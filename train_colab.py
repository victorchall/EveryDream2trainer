"""
Copyright [2022] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
import math
import signal
import argparse
import logging
import time
import gc
import random

import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

from colorama import Fore, Style, Cursor
import numpy as np
import itertools
import torch
import datetime
import json
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler
#from diffusers.models import AttentionBlock
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
#from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
from torch.utils.tensorboard import SummaryWriter

import keyboard

from data.every_dream import EveryDreamBatch
from utils.convert_diff_to_ckpt import convert as converter
from utils.gpu import GPU
forstepTime = time.time()

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

# def is_notebook() -> bool:
#     try:
#         from IPython import get_ipython
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter

def clean_filename(filename):
    """
    removes all non-alphanumeric characters from a string so it is safe to use as a filename
    """
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def convert_to_hf(ckpt_path):
    hf_cache = os.path.join("ckpt_cache", os.path.basename(ckpt_path))
    from utils.patch_unet import patch_unet

    if os.path.isfile(ckpt_path):        
        if not os.path.exists(hf_cache):
            os.makedirs(hf_cache)
            logging.info(f"Converting {ckpt_path} to Diffusers format")
            try:
                import utils.convert_original_stable_diffusion_to_diffusers as convert
                convert.convert(ckpt_path, f"ckpt_cache/{ckpt_path}")
            except:
                logging.info("Please manually convert the checkpoint to Diffusers format, see readme.")
                exit()
        else:
            logging.info(f"Found cached checkpoint at {hf_cache}")
        
        is_sd1attn = patch_unet(hf_cache)
        return hf_cache, is_sd1attn
    elif os.path.isdir(hf_cache):
        is_sd1attn = patch_unet(hf_cache)
        return hf_cache, is_sd1attn
    else:
        is_sd1attn = patch_unet(ckpt_path)
        return ckpt_path, is_sd1attn

def setup_local_logger(args):
    """
    configures logger with file and console logging, logs args, and returns the datestamp
    """
    log_path = args.logdir

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    json_config = json.dumps(vars(args), indent=2)
    datetimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(os.path.join(log_path, f"{args.project_name}-{datetimestamp}_cfg.json"), "w") as f:
        f.write(f"{json_config}")

    logfilename = os.path.join(log_path, f"{args.project_name}-{datetimestamp}.log")
    print(f" logging to {logfilename}")
    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                       )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    return datetimestamp

def log_optimizer(optimizer: torch.optim.Optimizer, betas, epsilon):
    """
    logs the optimizer settings
    """
    logging.info(f"{Fore.CYAN} * Optimizer: {optimizer.__class__.__name__} *{Style.RESET_ALL}")
    logging.info(f"    betas: {betas}, epsilon: {epsilon} *{Style.RESET_ALL}")

def save_optimizer(optimizer: torch.optim.Optimizer, path: str):
    """
    Saves the optimizer state
    """
    torch.save(optimizer.state_dict(), path)

def load_optimizer(optimizer, path: str):
    """
    Loads the optimizer state
    """
    optimizer.load_state_dict(torch.load(path))

def get_gpu_memory(nvsmi):
    """
    returns the gpu memory usage
    """
    gpu_query = nvsmi.DeviceQuery('memory.used, memory.total')
    gpu_used_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['used'])
    gpu_total_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['total'])
    return gpu_used_mem, gpu_total_mem

def append_epoch_log(global_step: int, epoch_pbar, gpu, log_writer, **logs):
    """
    updates the vram usage for the epoch
    """
    gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
    log_writer.add_scalar("performance/vram", gpu_used_mem, global_step)
    epoch_mem_color = Style.RESET_ALL
    if gpu_used_mem > 0.93 * gpu_total_mem:
        epoch_mem_color = Fore.LIGHTRED_EX
    elif gpu_used_mem > 0.85 * gpu_total_mem:
        epoch_mem_color = Fore.LIGHTYELLOW_EX
    elif gpu_used_mem > 0.7 * gpu_total_mem:
        epoch_mem_color = Fore.LIGHTGREEN_EX
    elif gpu_used_mem < 0.5 * gpu_total_mem:
        epoch_mem_color = Fore.LIGHTBLUE_EX

    if logs is not None:
        epoch_pbar.set_postfix(**logs, vram=f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step}")
        print(f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step} | Elapsed : {time.time() - forstepTime}s")


def set_args_12gb(args):
    logging.info(" Setting args to 12GB mode")
    if not args.gradient_checkpointing:   
        logging.info("  - Overiding gradient checkpointing to True")
        args.gradient_checkpointing = True
    if args.batch_size != 1:
        logging.info("  - Overiding batch size to 1")
        args.batch_size = 1
    # if args.grad_accum != 1:
    #     logging.info("   Overiding grad accum to 1")
        args.grad_accum = 1
    if args.resolution > 512:
        logging.info("  - Overiding resolution to 512")
        args.resolution = 512
    if not args.useadam8bit:
        logging.info("  - Overiding adam8bit to True")
        args.useadam8bit = True

def find_last_checkpoint(logdir):
    """
    Finds the last checkpoint in the logdir, recursively
    """
    last_ckpt = None
    last_date = None

    for root, dirs, files in os.walk(logdir):
        for file in files:
            if os.path.basename(file) == "model_index.json":
                curr_date = os.path.getmtime(os.path.join(root,file))

                if last_date is None or curr_date > last_date:
                    last_date = curr_date
                    last_ckpt = root

    assert last_ckpt, f"Could not find last checkpoint in logdir: {logdir}"
    assert "errored" not in last_ckpt, f"Found last checkpoint: {last_ckpt}, but it was errored, cancelling"

    print(f"    {Fore.LIGHTCYAN_EX}Found last checkpoint: {last_ckpt}, resuming{Style.RESET_ALL}")

    return last_ckpt

def setup_args(args):
    """
    Sets defaults for missing args (possible if missing from json config)
    Forces some args to be set based on others for compatibility reasons
    """
    if args.disable_unet_training and args.disable_textenc_training:
        raise ValueError("Both unet and textenc are disabled, nothing to train")

    if args.resume_ckpt == "findlast":
        logging.info(f"{Fore.LIGHTCYAN_EX} Finding last checkpoint in logdir: {args.logdir}{Style.RESET_ALL}")
        # find the last checkpoint in the logdir
        args.resume_ckpt = find_last_checkpoint(args.logdir)

    if args.lowvram:
        set_args_12gb(args)

    if not args.shuffle_tags:
        args.shuffle_tags = False

    args.clip_skip = max(min(4, args.clip_skip), 0)

    if args.ckpt_every_n_minutes is None and args.save_every_n_epochs is None:
        logging.info(f"{Fore.LIGHTCYAN_EX} No checkpoint saving specified, defaulting to every 20 minutes.{Style.RESET_ALL}")
        args.ckpt_every_n_minutes = 20

    if args.ckpt_every_n_minutes is None or args.ckpt_every_n_minutes < 1:
        args.ckpt_every_n_minutes = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs is None or args.save_every_n_epochs < 1:
        args.save_every_n_epochs = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs < _VERY_LARGE_NUMBER and args.ckpt_every_n_minutes < _VERY_LARGE_NUMBER:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** Both save_every_n_epochs and ckpt_every_n_minutes are set, this will potentially spam a lot of checkpoints{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** save_every_n_epochs: {args.save_every_n_epochs}, ckpt_every_n_minutes: {args.ckpt_every_n_minutes}{Style.RESET_ALL}")

    if args.cond_dropout > 0.26:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** cond_dropout is set fairly high: {args.cond_dropout}, make sure this was intended{Style.RESET_ALL}")

    if args.grad_accum > 1:
        logging.info(f"{Fore.CYAN} Batch size: {args.batch_size}, grad accum: {args.grad_accum}, 'effective' batch size: {args.batch_size * args.grad_accum}{Style.RESET_ALL}")

    total_batch_size = args.batch_size * args.grad_accum

    if args.scale_lr is not None and args.scale_lr:
        tmp_lr = args.lr
        args.lr = args.lr * (total_batch_size**0.55)
        logging.info(f"{Fore.CYAN} * Scaling learning rate {tmp_lr} by {total_batch_size**0.5}, new value: {args.lr}{Style.RESET_ALL}")

    if args.save_ckpt_dir is not None and not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    if args.rated_dataset:
        args.rated_dataset_target_dropout_percent = min(max(args.rated_dataset_target_dropout_percent, 0), 100)

        logging.info(logging.info(f"{Fore.CYAN} * Activating rated images learning with a target rate of {args.rated_dataset_target_dropout_percent}% {Style.RESET_ALL}"))

    return args

def main(args):
    """
    Main entry point
    """
    log_time = setup_local_logger(args)
    args = setup_args(args)

    if args.notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm.auto import tqdm

    seed = args.seed if args.seed != -1 else random.randint(0, 2**30)
    logging.info(f" Seed: {seed}")
    set_seed(seed)
    gpu = GPU()
    device = torch.device(f"cuda:{args.gpuid}")

    torch.backends.cudnn.benchmark = True

    log_folder = os.path.join(args.logdir, f"{args.project_name}_{log_time}")
    logging.info(f"Logging to {log_folder}")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    @torch.no_grad()
    def __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae, save_ckpt_dir, save_full_precision=False):
        """
        Save the model to disk
        """
        global global_step
        if global_step is None or global_step == 0:
            logging.warning("  No model to save, something likely blew up on startup, not saving")
            return
        logging.info(f" * Saving diffusers model to {save_path}")
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None, # save vram
            requires_safety_checker=None, # avoid nag
            feature_extractor=None, # must be none of no safety checker
        )
        pipeline.save_pretrained(save_path)
        sd_ckpt_path = f"{os.path.basename(save_path)}.ckpt"
        if save_ckpt_dir is not None:
            sd_ckpt_full = os.path.join(save_ckpt_dir, sd_ckpt_path)
        else:
            sd_ckpt_full = os.path.join(os.curdir, sd_ckpt_path)
        
        half = not save_full_precision

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=save_path, checkpoint_path=sd_ckpt_full, half=half)
        # optimizer_path = os.path.join(save_path, "optimizer.pt")

        # if self.save_optimizer_flag:
        #     logging.info(f" Saving optimizer state to {save_path}")
        #     self.save_optimizer(self.ctx.optimizer, optimizer_path)

    @torch.no_grad()
    def __create_inference_pipe(unet, text_encoder, tokenizer, scheduler, vae):
        """
        creates a pipeline for SD inference
        """
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None, # save vram
            requires_safety_checker=None, # avoid nag
            feature_extractor=None, # must be none of no safety checker
        )

        return pipe

    def __generate_sample(pipe: StableDiffusionPipeline, prompt : str, cfg: float, resolution: int, gen):
        """
        generates a single sample at a given cfg scale and saves it to disk
        """       
        with torch.no_grad(), autocast():
            image = pipe(prompt,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                    guidance_scale=cfg,
                    generator=gen,
                    height=resolution,
                    width=resolution,
            ).images[0]

            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype(font="arial.ttf", size=20)
            except:
                font = ImageFont.load_default()
            print_msg = f"cfg:{cfg:.1f}"

            l, t, r, b = draw.textbbox(xy=(0,0), text=print_msg, font=font)
            text_width = r - l
            text_height = b - t

            x = float(image.width - text_width - 10)
            y = float(image.height - text_height - 10)

            draw.rectangle((x, y, image.width, image.height), fill="white")
            draw.text((x, y), print_msg, fill="black", font=font)
        del draw, font
        return image

    def __generate_test_samples(pipe, prompts, gs, log_writer, log_folder, random_captions=False, resolution=512):      
        """
        generates samples at different cfg scales and saves them to disk
        """
        logging.info(f"Generating samples gs:{gs}, for {prompts}")
        seed = args.seed if args.seed != -1 else random.randint(0, 2**30)
        gen = torch.Generator(device=device).manual_seed(seed)

        i = 0
        for prompt in prompts:
            if prompt is None or len(prompt) < 2:
                #logging.warning("empty prompt in sample prompts, check your prompts file")
                continue
            images = []
            for cfg in [7.0, 4.0, 1.01]:
                image = __generate_sample(pipe, prompt, cfg, resolution=resolution, gen=gen)
                images.append(image)

            width = 0
            height = 0
            for image in images:
                width += image.width
                height = max(height, image.height)

            result = Image.new('RGB', (width, height))

            x_offset = 0
            for image in images:
                result.paste(image, (x_offset, 0))
                x_offset += image.width

            clean_prompt = clean_filename(prompt)

            result.save(f"{log_folder}/samples/gs{gs:05}-{i}-{clean_prompt[:100]}.jpg", format="JPEG", quality=95, optimize=True, progressive=False)
            with open(f"{log_folder}/samples/gs{gs:05}-{i}-{clean_prompt[:100]}.txt", "w", encoding='utf-8') as f:
                f.write(prompt)
                f.write(f"\n seed: {seed}")

            tfimage = transforms.ToTensor()(result)
            if random_captions:
                log_writer.add_image(tag=f"sample_{i}", img_tensor=tfimage, global_step=gs)
            else:
                log_writer.add_image(tag=f"sample_{i}_{clean_prompt[:100]}", img_tensor=tfimage, global_step=gs)
            i += 1

            del result
            del tfimage
            del images

    try: 
        hf_ckpt_path, is_sd1attn = convert_to_hf(args.resume_ckpt)
        text_encoder = CLIPTextModel.from_pretrained(hf_ckpt_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(hf_ckpt_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(hf_ckpt_path, subfolder="unet", upcast_attention=not is_sd1attn)
        sample_scheduler = DDIMScheduler.from_pretrained(hf_ckpt_path, subfolder="scheduler")
        noise_scheduler = DDPMScheduler.from_pretrained(hf_ckpt_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(hf_ckpt_path, subfolder="tokenizer", use_fast=False)
    except:
        logging.ERROR(" * Failed to load checkpoint *")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
    
    if not args.disable_xformers and (args.amp and is_sd1attn) or (not is_sd1attn):
        try:
            unet.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers")
        except Exception as ex:
            logging.warning("failed to load xformers, continuing without it")
            pass
    else:
        logging.info("xformers not available or disabled")

    default_lr = 2e-6
    curr_lr = args.lr if args.lr is not None else default_lr


    vae = vae.to(device, dtype=torch.float16 if args.amp else torch.float32)
    unet = unet.to(device, dtype=torch.float32)
    if args.disable_textenc_training and args.amp:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    else:
        text_encoder = text_encoder.to(device, dtype=torch.float32)

    if args.disable_textenc_training:
        logging.info(f"{Fore.CYAN} * NOT Training Text Encoder, quality reduced *{Style.RESET_ALL}")
        params_to_train = itertools.chain(unet.parameters())
    elif args.disable_unet_training:
        logging.info(f"{Fore.CYAN} * Training Text Encoder *{Style.RESET_ALL}")
        params_to_train = itertools.chain(text_encoder.parameters())
    else:
        logging.info(f"{Fore.CYAN} * Training Text Encoder *{Style.RESET_ALL}")
        params_to_train = itertools.chain(unet.parameters(), text_encoder.parameters())

    betas = (0.9, 0.999)
    epsilon = 1e-8
    if args.amp:
        epsilon = 2e-8
    
    weight_decay = 0.01
    if args.useadam8bit:
        import bitsandbytes as bnb
        opt_class = bnb.optim.AdamW8bit
        logging.info(f"{Fore.CYAN} * Using AdamW 8-bit Optimizer *{Style.RESET_ALL}")
    else:
        opt_class = torch.optim.AdamW
        logging.info(f"{Fore.CYAN} * Using AdamW standard Optimizer *{Style.RESET_ALL}")

    optimizer = opt_class(
            itertools.chain(params_to_train),
            lr=curr_lr,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
            amsgrad=False,
        )

    log_optimizer(optimizer, betas, epsilon)

    train_batch = EveryDreamBatch(
        data_root=args.data_root,
        flip_p=args.flip_p,
        debug_level=1,
        batch_size=args.batch_size,
        conditional_dropout=args.cond_dropout,
        resolution=args.resolution,
        tokenizer=tokenizer,
        seed = seed,
        log_folder=log_folder,
        write_schedule=args.write_schedule,
        shuffle_tags=args.shuffle_tags,
        rated_dataset=args.rated_dataset,
        rated_dataset_dropout_target=(1.0 - (args.rated_dataset_target_dropout_percent / 100.0))
    )

    torch.cuda.benchmark = False

    epoch_len = math.ceil(len(train_batch) / args.batch_size)

    if args.lr_decay_steps is None or args.lr_decay_steps < 1:
        args.lr_decay_steps = int(epoch_len * args.max_epochs * 1.5)

    lr_warmup_steps = int(args.lr_decay_steps / 50) if args.lr_warmup_steps is None else args.lr_warmup_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=args.lr_decay_steps,
    )

    sample_prompts = []
    with open(args.sample_prompts, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())


    if args.wandb is not None and args.wandb:
        wandb.init(project=args.project_name, sync_tensorboard=True, )
    else:
        log_writer = SummaryWriter(log_dir=log_folder, 
                                   flush_secs=5,
                                   comment="EveryDream2FineTunes",
                                  )

    def log_args(log_writer, args):
        arglog = "args:\n"
        for arg, value in sorted(vars(args).items()):
            arglog += f"{arg}={value}, "
        log_writer.add_text("config", arglog)
    
    log_args(log_writer, args)

    

    """
    Train the model

    """
    print(f" {Fore.LIGHTGREEN_EX}** Welcome to EveryDream trainer 2.0!**{Style.RESET_ALL}")
    print(f" (C) 2022 Victor C Hall  This program is licensed under AGPL 3.0 https://www.gnu.org/licenses/agpl-3.0.en.html")
    print()
    print("** Trainer Starting **")

    global interrupted
    interrupted = False

    def sigterm_handler(signum, frame):
        """
        handles sigterm
        """
        global interrupted
        if not interrupted:
            interrupted=True            
            global global_step
            #TODO: save model on ctrl-c
            interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step}")
            print()
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            time.sleep(2) # give opportunity to ctrl-C again to cancel save
            __save_model(interrupted_checkpoint_path, unet, text_encoder, tokenizer, noise_scheduler, vae, args.save_ckpt_dir, args.save_full_precision)
        exit(_SIGTERM_EXIT_CODE)

    signal.signal(signal.SIGINT, sigterm_handler)
    
    if not os.path.exists(f"{log_folder}/samples/"):
        os.makedirs(f"{log_folder}/samples/")

    gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
    logging.info(f" Pretraining GPU Memory: {gpu_used_mem} / {gpu_total_mem} MB")
    logging.info(f" saving ckpts every {args.ckpt_every_n_minutes} minutes")
    logging.info(f" saving ckpts every {args.save_every_n_epochs } epochs")


    def collate_fn(batch):
        """
        Collates batches
        """
        images = [example["image"] for example in batch]
        captions = [example["caption"] for example in batch]
        tokens = [example["tokens"] for example in batch]
        runt_size = batch[0]["runt_size"]

        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()

        ret = {
            "tokens": torch.stack(tuple(tokens)),
            "image": images,
            "captions": captions,
            "runt_size": runt_size,
        }
        del batch
        return ret

    train_dataloader = torch.utils.data.DataLoader(
        train_batch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    unet.train() if not args.disable_unet_training else unet.eval()
    text_encoder.train() if not args.disable_textenc_training else text_encoder.eval() 

    logging.info(f" unet device: {unet.device}, precision: {unet.dtype}, training: {unet.training}")
    logging.info(f" text_encoder device: {text_encoder.device}, precision: {text_encoder.dtype}, training: {text_encoder.training}")
    logging.info(f" vae device: {vae.device}, precision: {vae.dtype}, training: {vae.training}")
    logging.info(f" scheduler: {noise_scheduler.__class__}")

    logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.grad_accum}{Style.RESET_ALL}"), 
    logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")


    #scaler = torch.cuda.amp.GradScaler()
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp,
        #enabled=True,
        init_scale=2**17.5,
        growth_factor=1.8,
        backoff_factor=1.0/1.8,
        growth_interval=50,
    )
    logging.info(f" Grad scaler enabled: {scaler.is_enabled()} (amp mode)")


    epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")

    # steps_pbar = tqdm(range(epoch_len), position=1, leave=True)
    # steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

    epoch_times = []

    global global_step
    global_step = 0
    training_start_time = time.time()
    last_epoch_saved_time = training_start_time

    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    loss_log_step = []
    
    try:
        for epoch in range(args.max_epochs):
            loss_epoch = []
            epoch_start_time = time.time()
            images_per_sec_log_step = []

            epoch_len = math.ceil(len(train_batch) / args.batch_size)
            steps_pbar = tqdm(range(epoch_len), position=1)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

            for step, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                with torch.no_grad():
                    with autocast(enabled=args.amp):     
                        pixel_values = batch["image"].to(memory_format=torch.contiguous_format).to(unet.device)
                        latents = vae.encode(pixel_values, return_dict=False)
                    del pixel_values
                    latents = latents[0].sample() * 0.18215

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    cuda_caption = batch["tokens"].to(text_encoder.device)

                #with autocast(enabled=args.amp):
                encoder_hidden_states = text_encoder(cuda_caption, output_hidden_states=True)

                if args.clip_skip > 0:
                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states.hidden_states[-args.clip_skip])
                else:
                    encoder_hidden_states = encoder_hidden_states.last_hidden_state

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                del noise, latents, cuda_caption

                with autocast(enabled=args.amp):
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                del timesteps, encoder_hidden_states, noisy_latents
                #with autocast(enabled=args.amp):
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                #if args.amp:
                scaler.scale(loss).backward()
                #else:
                #    loss.backward()

                if args.clip_grad_norm is not None:
                    if not args.disable_unet_training:
                        torch.nn.utils.clip_grad_norm_(parameters=unet.parameters(), max_norm=args.clip_grad_norm)
                    if not args.disable_textenc_training:
                        torch.nn.utils.clip_grad_norm_(parameters=text_encoder.parameters(), max_norm=args.clip_grad_norm)

                if batch["runt_size"] > 0:
                    grad_scale = batch["runt_size"] / args.batch_size
                    with torch.no_grad(): # not required? just in case for now, needs more testing
                        for param in unet.parameters():
                            if param.grad is not None:
                                param.grad *= grad_scale
                        if text_encoder.training:
                            for param in text_encoder.parameters():
                                if param.grad is not None:
                                    param.grad *= grad_scale

                if ((global_step + 1) % args.grad_accum == 0) or (step == epoch_len - 1):
                    # if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                    # else:
                    #     optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step()

                loss_step = loss.detach().item()

                steps_pbar.set_postfix({"loss/step": loss_step},{"gs": global_step})
                steps_pbar.update(1)

                images_per_sec = args.batch_size / (time.time() - step_start_time)
                images_per_sec_log_step.append(images_per_sec)

                loss_log_step.append(loss_step)
                loss_epoch.append(loss_step)

                if (global_step + 1) % args.log_step == 0:
                    curr_lr = lr_scheduler.get_last_lr()[0]
                    loss_local = sum(loss_log_step) / len(loss_log_step)
                    loss_log_step = []
                    logs = {"loss/log_step": loss_local, "lr": curr_lr, "img/s": images_per_sec}                    
                    log_writer.add_scalar(tag="hyperparamater/lr", scalar_value=curr_lr, global_step=global_step)
                    log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_local, global_step=global_step)
                    sum_img = sum(images_per_sec_log_step)
                    avg = sum_img / len(images_per_sec_log_step)
                    images_per_sec_log_step = []
                    if args.amp:
                        log_writer.add_scalar(tag="hyperparamater/grad scale", scalar_value=scaler.get_scale(), global_step=global_step)
                    log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)
                    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                    torch.cuda.empty_cache()

                if (not args.notebook and keyboard.is_pressed("ctrl+alt+page up")) or ((global_step + 1) % args.sample_steps == 0):
                    pipe = __create_inference_pipe(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=sample_scheduler, vae=vae)
                    pipe = pipe.to(device)

                    with torch.no_grad():
                        if sample_prompts is not None and len(sample_prompts) > 0 and len(sample_prompts[0]) > 1:
                            __generate_test_samples(pipe=pipe, prompts=sample_prompts, log_writer=log_writer, log_folder=log_folder, gs=global_step, resolution=args.resolution)
                        else:
                            max_prompts = min(4,len(batch["captions"]))
                            prompts=batch["captions"][:max_prompts]
                            __generate_test_samples(pipe=pipe, prompts=prompts, log_writer=log_writer, log_folder=log_folder, gs=global_step, random_captions=True, resolution=args.resolution)

                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()

                min_since_last_ckpt =  (time.time() - last_epoch_saved_time) / 60

                if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                    last_epoch_saved_time = time.time()
                    logging.info(f"Saving model, {args.ckpt_every_n_minutes} mins at step {global_step}")
                    save_path = os.path.join(f"{log_folder}/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")
                    __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, args.save_ckpt_dir, args.save_full_precision)

                if epoch > 0 and epoch % args.save_every_n_epochs == 0 and step == 1 and epoch < args.max_epochs - 1:
                    logging.info(f" Saving model, {args.save_every_n_epochs} epochs at step {global_step}")
                    save_path = os.path.join(f"{log_folder}/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")
                    __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, args.save_ckpt_dir, args.save_full_precision)

                del batch
                global_step += 1

                if global_step == 500:
                    scaler.set_growth_factor(1.4)
                    scaler.set_backoff_factor(1/1.4)
                if global_step == 1000:
                    scaler.set_growth_factor(1.2)
                    scaler.set_backoff_factor(1/1.2)
                    scaler.set_growth_interval(100)
                # end of step

            steps_pbar.close()

            elapsed_epoch_time = (time.time() - epoch_start_time) / 60
            epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
            log_writer.add_scalar("performance/minutes per epoch", elapsed_epoch_time, global_step)

            epoch_pbar.update(1)
            if epoch < args.max_epochs - 1:
                train_batch.shuffle(epoch_n=epoch, max_epochs = args.max_epochs)

            loss_local = sum(loss_epoch) / len(loss_epoch)
            log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_local, global_step=global_step)
            # end of epoch

        # end of training

        save_path = os.path.join(f"{log_folder}/ckpts/last-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, args.save_ckpt_dir, args.save_full_precision)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        save_path = os.path.join(f"{log_folder}/ckpts/errored-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, args.save_ckpt_dir, args.save_full_precision)
        raise ex

    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")


def update_old_args(t_args):
    """
    Update old args to new args to deal with json config loading and missing args for compatibility
    """
    if not hasattr(t_args, "shuffle_tags"):
        print(f" Config json is missing 'shuffle_tags' flag")
        t_args.__dict__["shuffle_tags"] = False
    if not hasattr(t_args, "save_full_precision"):
        print(f" Config json is missing 'save_full_precision' flag")
        t_args.__dict__["save_full_precision"] = False
    if not hasattr(t_args, "notebook"):
        print(f" Config json is missing 'notebook' flag")
        t_args.__dict__["notebook"] = False
    if not hasattr(t_args, "disable_unet_training"):
        print(f" Config json is missing 'disable_unet_training' flag")
        t_args.__dict__["disable_unet_training"] = False
    if not hasattr(t_args, "rated_dataset"):
        print(f" Config json is missing 'rated_dataset' flag")
        t_args.__dict__["rated_dataset"] = False
    if not hasattr(t_args, "rated_dataset_target_dropout_percent"):
        print(f" Config json is missing 'rated_dataset_target_dropout_percent' flag")
        t_args.__dict__["rated_dataset_target_dropout_percent"] = 50


if __name__ == "__main__":
    supported_resolutions = [256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152]
    supported_precisions = ['fp16', 'fp32']
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, _ = argparser.parse_known_args()

    if args.config is not None:
        print(f"Loading training config from {args.config}, all other command options will be ignored!")
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            update_old_args(t_args) # update args to support older configs
            print(f" args: \n{t_args.__dict__}")
            args = argparser.parse_args(namespace=t_args)
    else:
        print("No config file specified, using command line args")
        argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
        argparser.add_argument("--amp", action="store_true", default=False, help="Enables automatic mixed precision compute, recommended on")
        argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
        argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
        argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
        argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
        argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
        argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
        argparser.add_argument("--disable_textenc_training", action="store_true", default=False, help="disables training of text encoder (def: False)")
        argparser.add_argument("--disable_unet_training", action="store_true", default=False, help="disables training of unet (def: False) NOT RECOMMENDED")
        argparser.add_argument("--disable_xformers", action="store_true", default=False, help="disable xformers, may reduce performance (def: False)")
        argparser.add_argument("--flip_p", type=float, default=0.0, help="probability of flipping image horizontally (def: 0.0) use 0.0 to 1.0, ex 0.5, not good for specific faces!")
        argparser.add_argument("--gpuid", type=int, default=0, help="id of gpu to use for training, (def: 0) (ex: 1 to use GPU_ID 1)")
        argparser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="enable gradient checkpointing to reduce VRAM use, may reduce performance (def: False)")
        argparser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation factor (def: 1), (ex, 2)")
        argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
        argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default!")
        argparser.add_argument("--lowvram", action="store_true", default=False, help="automatically overrides various args to support 12GB gpu")
        argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
        argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
        argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
        argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant")
        argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
        argparser.add_argument("--notebook", action="store_true", default=False, help="disable keypresses and uses tqdm.notebook for jupyter notebook (def: False)")
        argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
        argparser.add_argument("--resolution", type=int, default=512, help="resolution to train", choices=supported_resolutions)
        argparser.add_argument("--resume_ckpt", type=str, required=True, default="sd_v1-5_vae.ckpt")
        argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt", help="File with prompts to generate test samples from (def: sample_prompts.txt)")
        argparser.add_argument("--sample_steps", type=int, default=250, help="Number of steps between samples (def: 250)")
        argparser.add_argument("--save_ckpt_dir", type=str, default=None, help="folder to save checkpoints to (def: root training folder)")
        argparser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every n epochs, def: 0 (disabled)")
        argparser.add_argument("--save_full_precision", action="store_true", default=False, help="save ckpts at full FP32")
        argparser.add_argument("--save_optimizer", action="store_true", default=False, help="saves optimizer state with ckpt, useful for resuming training later")
        argparser.add_argument("--scale_lr", action="store_true", default=False, help="automatically scale up learning rate based on batch size and grad accumulation (def: False)")
        argparser.add_argument("--seed", type=int, default=555, help="seed used for samples and shuffling, use -1 for random")
        argparser.add_argument("--shuffle_tags", action="store_true", default=False, help="randomly shuffles CSV tags in captions, for booru datasets")
        argparser.add_argument("--useadam8bit", action="store_true", default=False, help="Use AdamW 8-Bit optimizer, recommended!")
        argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
        argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
        argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
        argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")

        args, _ = argparser.parse_known_args()

    main(args)
