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

import torch.nn.functional as torch_functional
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

from colorama import Fore, Style, Cursor
import numpy as np
import itertools
import torch
import datetime
import json
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline, DDPMScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler
#from diffusers.models import AttentionBlock
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
#from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import EveryDreamBatch
from utils.convert_diffusers_to_stable_diffusion import convert as converter
from utils.gpu import GPU

_GRAD_ACCUM_STEPS = 1 # future use...
_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

def convert_to_hf(ckpt_path):
    hf_cache = os.path.join("ckpt_cache", os.path.basename(ckpt_path))

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
        return hf_cache
    elif os.path.isdir(hf_cache):
        return hf_cache
    else:
        return ckpt_path

def setup_local_logger(args):
    """
    configures logger with file and console logging, logs args, and returns the datestamp
    """
    log_path = args.logdir
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    json_config = json.dumps(vars(args), indent=2)
    datetimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logfilename = os.path.join(log_path, f"{args.project_name}-train{datetimestamp}.log")
    with open(logfilename, "w") as f:
        f.write(f"Training config:\n{json_config}\n")

    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                       )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return datetimestamp

def log_optimizer(optimizer: torch.optim.Optimizer, betas, epsilon):
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


def main(args):
    """
    Main entry point
    """
    log_time = setup_local_logger(args)

    seed = 555
    set_seed(seed)
    gpu = GPU()
    
    if args.ckpt_every_n_minutes is None and args.save_every_n_epochs is None:
        logging.info(" no checkpointing specified, defaulting to 20 minutes")
        args.ckpt_every_n_minutes = 20

    if args.ckpt_every_n_minutes is None or args.ckpt_every_n_minutes < 1:
        args.ckpt_every_n_minutes = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs is None or args.save_every_n_epochs < 1:
        args.save_every_n_epochs = _VERY_LARGE_NUMBER
    
    if args.save_every_n_epochs < _VERY_LARGE_NUMBER and args.ckpt_every_n_minutes < _VERY_LARGE_NUMBER:
        logging.warning(f"{Fore.YELLOW}Both save_every_n_epochs and ckpt_every_n_minutes are set, this will potentially spam a lot of checkpoints{Style.RESET_ALL}")
        logging.warning(f"{Fore.YELLOW}save_every_n_epochs: {args.save_every_n_epochs}, ckpt_every_n_minutes: {args.ckpt_every_n_minutes}{Style.RESET_ALL}")

    if args.cond_dropout > 0.26:
        logging.warning(f"{Fore.YELLOW}cond_dropout is set fairly high: {args.cond_dropout}, make sure this was intended{Style.RESET_ALL}")


    @torch.no_grad()
    def __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae):
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
        sd_ckpt_full = os.path.join(os.curdir, sd_ckpt_path)

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=save_path, checkpoint_path=sd_ckpt_full, half=True)
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
        if is_xformers_available():
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as ex:
                print("failed to load xformers, continuing without it")
                pass
        return pipe

    def __generate_sample(pipe: StableDiffusionPipeline, prompt : str, cfg: float, resolution: int):
        """
        generates a single sample at a given cfg scale and saves it to disk
        """
        gen = torch.Generator(device="cuda").manual_seed(555)
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
            font = ImageFont.truetype(size=24)
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

    #@torch.no_grad()
    def __generate_test_samples(pipe, prompts, gs, log_writer, log_folder, random_captions=False, resolution=512):      
        """
        generates samples at different cfg scales and saves them to disk
        """
        logging.info(f"Generating samples gs:{gs}, for {prompts}")

        #with torch.inference_mode(), suppress_stdout():
        #with autocast():
        i = 0
        for prompt in prompts:
            if prompt is None or len(prompt) < 2:
                logging.warning("empty prompt in sample prompts, check your prompts file")
                continue
            images = []
            for cfg in [7.0, 4.0, 1.01]:
                image = __generate_sample(pipe, prompt, cfg, resolution=resolution)
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

            result.save(f"{log_folder}/samples/gs{gs:05}-{prompt[:100]}.png")

            tfimage = transforms.ToTensor()(result)
            if random_captions:
                log_writer.add_image(tag=f"sample_{i}", img_tensor=tfimage, global_step=gs)
                i += 1
            else:
                log_writer.add_image(tag=f"sample_{prompt[:150]}", img_tensor=tfimage, global_step=gs)

            del result
            del tfimage
            del images

    try: 
        hf_ckpt_path = convert_to_hf(args.resume_ckpt)
        text_encoder = CLIPTextModel.from_pretrained(hf_ckpt_path, subfolder="text_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKL.from_pretrained(hf_ckpt_path, subfolder="vae", torch_dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(hf_ckpt_path, subfolder="unet", torch_dtype=torch.float32)
        scheduler = DDIMScheduler.from_pretrained(hf_ckpt_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(hf_ckpt_path, subfolder="tokenizer", use_fast=False)
    except:
        logging.ERROR(" * Failed to load checkpoint *")

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logging.info(" Enabled memory efficient attention (xformers)")
        except Exception as e:
            logging.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    default_lr = 2e-6 if args.useadam8bit else 2e-6
    lr = args.lr if args.lr is not None else default_lr

    vae = vae.to(torch.device("cuda"), dtype=torch.float32)
    unet = unet.to(torch.device("cuda"))
    text_encoder = text_encoder.to(torch.device("cuda"))

    if args.disable_textenc_training:
        logging.info(f"{Fore.CYAN} * NOT Training Text Encoder, quality reduced *{Style.RESET_ALL}")
        params_to_train = itertools.chain(unet.parameters())
        text_encoder.eval()
    else:
        logging.info(f"{Fore.CYAN} * Training Text Encoder *{Style.RESET_ALL}")
        params_to_train = itertools.chain(unet.parameters(), text_encoder.parameters())

    betas = (0.9, 0.999)
    epsilon = 1e-8 if args.mixed_precision == "NO" else 1e-7
    weight_decay = 0.01
    if args.useadam8bit:
        logging.info(f"{Fore.CYAN} * Using AdamW 8-bit Optimizer *{Style.RESET_ALL}")
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            itertools.chain(params_to_train),
            lr=lr,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
    else:
        logging.info(f"{Fore.CYAN} * Using AdamW8 standard Optimizer *{Style.RESET_ALL}")
        optimizer = torch.optim.AdamW(
            itertools.chain(params_to_train),
            lr=lr,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
            amsgrad=False,
        )

    log_optimizer(optimizer, betas, epsilon)

    train_batch = EveryDreamBatch(
        data_root=args.data_root,
        flip_p=0.0,
        debug_level=1,
        batch_size=args.batch_size,
        conditional_dropout=args.cond_dropout,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )

    torch.cuda.benchmark = False

    epoch_len = math.ceil(len(train_batch) / args.batch_size)

    if args.lr_decay_steps is None or args.lr_decay_steps < 1:
        args.lr_decay_steps = int(epoch_len * args.max_epochs * 1.2)

    lr_warmup_steps = int(args.lr_decay_steps / 50) if args.lr_warmup_steps is None else args.lr_warmup_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * _GRAD_ACCUM_STEPS,
        num_training_steps=args.lr_decay_steps * _GRAD_ACCUM_STEPS,
    )

    # read prompts from prompts_file.txt
    sample_prompts = []
    with open(args.sample_prompts, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())

    log_folder = os.path.join("logs", f"{args.project_name}{log_time}")

    if False: #args.wandb is not None and args.wandb: # not yet supported
        log_writer = wandb.init(project="EveryDream2FineTunes", 
                                name=args.project_name, 
                                dir=log_folder,
                               )
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

    args.clip_skip = max(min(2, args.clip_skip), 0)

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
            interrupted_checkpoint_path = os.path.join(f"{log_folder}/interrupted-gs{global_step}")
            print()
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            __save_model(interrupted_checkpoint_path, unet, text_encoder, tokenizer, scheduler, vae)
        exit(_SIGTERM_EXIT_CODE)

    signal.signal(signal.SIGINT, sigterm_handler)
    
    if not os.path.exists(f"{log_folder}/samples/"):
        os.makedirs(f"{log_folder}/samples/")

    gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
    logging.info(f" Pretraining GPU Memory: {gpu_used_mem} / {gpu_total_mem} MB")
    logging.info(f" saving ckpts every {args.ckpt_every_n_minutes} minutes")
    logging.info(f" saving ckpts every {args.save_every_n_epochs } epochs")

    scaler = torch.cuda.amp.GradScaler(
        enabled=False,
        #enabled=True if args.sd1 else False,
        init_scale=2**16,
        growth_factor=1.000001,
        backoff_factor=0.9999999,
        growth_interval=50,
    )
    logging.info(f" Grad scaler enabled: {scaler.is_enabled()}")

    def collate_fn(batch):
        """
        Collates batches
        """
        images = [example["image"] for example in batch]
        captions = [example["caption"] for example in batch]
        tokens = [example["tokens"] for example in batch]

        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()

        batch = {
            "tokens": torch.stack(tuple(tokens)),
            "image": images,
            "captions": captions,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_batch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    total_batch_size = args.batch_size * _GRAD_ACCUM_STEPS
    

    unet.train()
    text_encoder.requires_grad_(True)
    text_encoder.train()    

    logging.info(f" unet device: {unet.device}, precision: {unet.dtype}, training: {unet.training}")
    logging.info(f" text_encoder device: {text_encoder.device}, precision: {text_encoder.dtype}, training: {text_encoder.training}")
    logging.info(f" vae device: {vae.device}, precision: {vae.dtype}, training: {vae.training}")
    logging.info(f" scheduler: {scheduler.__class__}")

    logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{_GRAD_ACCUM_STEPS}{Style.RESET_ALL}"), 
    logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
    #logging.info(f" {Fore.GREEN}total_batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{total_batch_size}")
    logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")

    epoch_pbar = tqdm(range(args.max_epochs), position=0)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")

    steps_pbar = tqdm(range(epoch_len), position=1)
    steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

    epoch_times = []

    global global_step
    global_step = 0
    training_start_time = time.time()
    last_epoch_saved_time = training_start_time

    # (global_step: int, epoch_pbar, gpu, log_writer, **logs):
    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    torch.cuda.empty_cache()

    try:            
        for epoch in range(args.max_epochs):
            if epoch > 0 and epoch % args.save_every_n_epochs == 0:
                logging.info(f" Saving model")
                save_path = os.path.join(f"logs/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")
                __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae)

            epoch_start_time = time.time()
            steps_pbar.reset()
            images_per_sec_epoch = []

            #for step, batch in enumerate(self.ctx.train_dataloader):
            for step, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                with torch.no_grad():
                    with autocast():
                        pixel_values = batch["image"].to(memory_format=torch.contiguous_format).to(unet.device)
                        latents = vae.encode(pixel_values, return_dict=False)

                    latent = latents[0]
                    latents = latent.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                cuda_caption = batch["tokens"].to(text_encoder.device)  

                encoder_hidden_states = text_encoder(cuda_caption)

                # if clip_skip > 0: #TODO
                #     encoder_hidden_states = encoder_hidden_states['last_hidden_state'][-clip_skip]
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
                #del noise, latents

                #with torch.cuda.amp.autocast(enabled=lowvram):
                with autocast(): # xformers requires fp16
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states.last_hidden_state).sample

                with autocast(enabled=args.sd1):
                    loss = torch_functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                #del timesteps, encoder_hidden_states, noisy_latents

                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters=unet.parameters(), max_norm=args.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(parameters=text_encoder.parameters(), max_norm=args.clip_grad_norm)
                
                #with torch.cuda.amp(enabled=False):
                #if args.mixed_precision in ['bf16','fp16']:
                if args.sd1:
                    with autocast():
                        scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                steps_pbar.update(1)
                global_step += 1

                images_per_sec = args.batch_size / (time.time() - step_start_time)
                images_per_sec_epoch.append(images_per_sec)

                #with torch.no_grad():
                if (global_step + 1) % args.log_step == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    logs = {"loss/step": loss.detach().item(), "lr": lr, "img/s": images_per_sec, "scale": scaler.get_scale()}
                    log_writer.add_scalar(tag="loss/step", scalar_value=loss, global_step=global_step)
                    log_writer.add_scalar(tag="hyperparamater/lr", scalar_value=lr, global_step=global_step)
                    sum_img = sum(images_per_sec_epoch)
                    avg = sum_img / len(images_per_sec_epoch)
                    images_per_sec_epoch = []
                    #log_writer.add_scalar(tag="hyperparamater/grad scale", scalar_value=scaler.get_scale(), global_step=global_step)
                    log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)
                    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)

                if (global_step + 1) % args.sample_steps == 0:
                    #(unet, text_encoder, tokenizer, scheduler):
                    pipe = __create_inference_pipe(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, vae=vae)
                    pipe = pipe.to(torch.device("cuda"))

                    with torch.no_grad():
                        if sample_prompts is not None and len(sample_prompts) > 0 and len(sample_prompts[0]) > 1:
                            #(pipe, prompts, gs, log_writer, log_folder, random_captions=False):
                            __generate_test_samples(pipe=pipe, prompts=sample_prompts, log_writer=log_writer, log_folder=log_folder, gs=global_step, resolution=args.resolution)
                        else:
                            max_prompts = min(4,len(batch["captions"]))
                            prompts=batch["captions"][:max_prompts]
                            __generate_test_samples(pipe=pipe, prompts=prompts, log_writer=log_writer, log_folder=log_folder, gs=global_step, random_captions=True)

                    del pipe
                    torch.cuda.empty_cache()

                min_since_last_ckpt =  (time.time() - last_epoch_saved_time) /  60

                if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                    last_epoch_saved_time = time.time()
                    logging.info(f"Saving model at {args.ckpt_every_n_minutes} mins at step {global_step}")
                    save_path = os.path.join(f"{log_folder}/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")

                    __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae)

                # end of step

            # end of epoch
            elapsed_epoch_time = (time.time() - epoch_start_time) / 60         
            epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
            log_writer.add_scalar("performance/minutes per epoch", elapsed_epoch_time, global_step)

            epoch_pbar.update(1)

        # end of training

        save_path = os.path.join(f"{log_folder}/ckpts/last-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time:.2f} seconds, total steps: {global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]) / 60:.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        save_path = os.path.join(f"{log_folder}/ckpts/errored-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae)
        raise ex

    logging.info(f"{Fore.LIGHTWHITE_EX} *Finished training *{Style.RESET_ALL}")


if __name__ == "__main__":
    supported_resolutions = [512, 576, 640, 704, 768, 832, 896, 960, 1024]
    argparser = argparse.ArgumentParser(description="EveryDream Training options")
    argparser.add_argument("--resume_ckpt", type=str, required=True, default="sd_v1-5_vae.ckpt")
    argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant scheduler")
    argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default")
    argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every n epochs, def: 0 (disabled)")
    argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--useadam8bit", action="store_true", default=False, help="Use AdamW 8-Bit optimizer")
    argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
    argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt", help="File with prompts to generate test samples from (def: sample_prompts.txt)")
    argparser.add_argument("--sample_steps", type=int, default=250, help="Number of steps between samples (def: 250)")
    argparser.add_argument("--disable_textenc_training", action="store_true", default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--grad_accum", type=int, default=1, help="NONFUNCTIONING. Gradient accumulation factor (def: 1), (ex, 2)")
    argparser.add_argument("--clip_skip", type=int, default=0, help="NONFUNCTIONING. Train using penultimate layers (def: 0)", choices=[0, 1, 2])
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--mixed_precision", default="no", help="NONFUNCTIONING. precision, (default: NO for fp32)", choices=["NO", "fp16", "bf16"])
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--save_optimizer", action="store_true", default=False, help="saves optimizer state with ckpt, useful for resuming training later")
    argparser.add_argument("--resolution", type=int, default=512, help="resolution to train", choices=supported_resolutions)
    argparser.add_argument("--sd1", action="store_true", default=False, help="set if training SD1.x, else SD2 is assumed")
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
    argparser.add_argument("--save_ckpt_dir", type=str, default=None, help="folder to save checkpoints to (def: root training folder)")
    args = argparser.parse_args()

    main(args)
