"""
Copyright [2022-2023] Victor C Hall

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
import pprint
import sys
import math
import signal
import argparse
import logging
import threading
import time
import gc
import random
import traceback
import shutil
import importlib

import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from colorama import Fore, Style
import numpy as np
import itertools
import torch
import datetime
import json

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler, \
    DPMSolverMultistepScheduler
#from diffusers.models import AttentionBlock
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
#from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import webbrowser
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import DataLoaderMultiAspect

from data.every_dream import EveryDreamBatch, build_torch_dataloader
from data.every_dream_validation import EveryDreamValidator
from data.image_train_item import ImageTrainItem
from utils.huggingface_downloader import try_download_model_from_hf
from utils.convert_diff_to_ckpt import convert as converter
from utils.isolate_rng import isolate_rng

if torch.cuda.is_available():
    from utils.gpu import GPU
import data.aspects as aspects
import data.resolver as resolver
from utils.sample_generator import SampleGenerator

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

def get_hf_ckpt_cache_path(ckpt_path):
    return os.path.join("ckpt_cache", os.path.basename(ckpt_path))

def convert_to_hf(ckpt_path):

    hf_cache = get_hf_ckpt_cache_path(ckpt_path)
    from utils.analyze_unet import get_attn_yaml

    if os.path.isfile(ckpt_path):
        if not os.path.exists(hf_cache):
            os.makedirs(hf_cache)
            logging.info(f"Converting {ckpt_path} to Diffusers format")
            try:
                import utils.convert_original_stable_diffusion_to_diffusers as convert
                convert.convert(ckpt_path, f"ckpt_cache/{ckpt_path}")
            except:
                logging.info("Please manually convert the checkpoint to Diffusers format (one time setup), see readme.")
                exit()
        else:
            logging.info(f"Found cached checkpoint at {hf_cache}")

        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    elif os.path.isdir(hf_cache):
        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    else:
        is_sd1attn, yaml = get_attn_yaml(ckpt_path)
        return ckpt_path, is_sd1attn, yaml

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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(lambda msg: "Palette images with Transparency expressed in bytes" in msg.getMessage())
    logging.getLogger().addHandler(console_handler)
    import warnings
    warnings.filterwarnings("ignore", message="UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images")
    #from PIL import Image

    return datetimestamp

def log_optimizer(optimizer: torch.optim.Optimizer, betas, epsilon, weight_decay, unet_lr, text_encoder_lr):
    """
    logs the optimizer settings
    """
    logging.info(f"{Fore.CYAN} * Optimizer: {optimizer.__class__.__name__} *{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}    unet lr: {unet_lr}, text encoder lr: {text_encoder_lr}, betas: {betas}, epsilon: {epsilon}, weight_decay: {weight_decay} *{Style.RESET_ALL}")

def save_optimizer(optimizer: torch.optim.Optimizer, path: str):
    """
    Saves the optimizer state
    """
    torch.save(optimizer.state_dict(), path)

def load_optimizer(optimizer: torch.optim.Optimizer, path: str):
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
    if gpu is not None:
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

def set_args_12gb(args):
    logging.info(" Setting args to 12GB mode")
    if not args.gradient_checkpointing:
        logging.info("  - Overiding gradient checkpointing to True")
        args.gradient_checkpointing = True
    if args.batch_size > 2:
        logging.info("  - Overiding batch size to max 2")
        args.batch_size = 2
        args.grad_accum = 1
    if args.resolution > 512:
        logging.info("  - Overiding resolution to max 512")
        args.resolution = 512

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
    if args.disable_amp:
        logging.warning(f"{Fore.LIGHTYELLOW_EX} Disabling AMP, not recommended.{Style.RESET_ALL}")
        args.amp = False
    else:
        args.amp = True

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

    if args.useadam8bit:
        logging.warning(f"{Fore.LIGHTYELLOW_EX} Useadam8bit arg is deprecated, use optimizer.json instead, which defaults to useadam8bit anyway{Style.RESET_ALL}")

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

    args.aspects = aspects.get_aspect_buckets(args.resolution)

    return args

def update_grad_scaler(scaler: GradScaler, global_step, epoch, step):
    if global_step == 500:
        factor = 1.8
        scaler.set_growth_factor(factor)
        scaler.set_backoff_factor(1/factor)
        scaler.set_growth_interval(50)
    if global_step == 1000:
        factor = 1.6
        scaler.set_growth_factor(factor)
        scaler.set_backoff_factor(1/factor)
        scaler.set_growth_interval(50)
    if global_step == 2000:
        factor = 1.3
        scaler.set_growth_factor(factor)
        scaler.set_backoff_factor(1/factor)
        scaler.set_growth_interval(100)
    if global_step == 4000:
        factor = 1.15
        scaler.set_growth_factor(factor)
        scaler.set_backoff_factor(1/factor)
        scaler.set_growth_interval(100)

def report_image_train_item_problems(log_folder: str, items: list[ImageTrainItem]) -> None:
    for item in items:
        if item.error is not None:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {item.error}")

    undersized_items = [item for item in items if item.is_undersized]

    if len(undersized_items) > 0:
        underized_log_path = os.path.join(log_folder, "undersized_images.txt")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Some images are smaller than the target size, consider using larger images{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Check {underized_log_path} for more information.{Style.RESET_ALL}")
        with open(underized_log_path, "w", encoding='utf-8') as undersized_images_file:
            undersized_images_file.write(f" The following images are smaller than the target size, consider removing or sourcing a larger copy:")
            for undersized_item in undersized_items:
                message = f" *** {undersized_item.pathname} with size: {undersized_item.image_size} is smaller than target size: {undersized_item.target_wh}\n"
                undersized_images_file.write(message)

def resolve_image_train_items(args: argparse.Namespace, log_folder: str) -> list[ImageTrainItem]:
    logging.info(f"* DLMA resolution {args.resolution}, buckets: {args.aspects}")
    logging.info(" Preloading images...")

    resolved_items = resolver.resolve(args.data_root, args)
    report_image_train_item_problems(log_folder, resolved_items)
    image_paths = set(map(lambda item: item.pathname, resolved_items))

    # Remove erroneous items
    image_train_items = [item for item in resolved_items if item.error is None]
    print (f" * Found {len(image_paths)} files in '{args.data_root}'")

    return image_train_items

def write_batch_schedule(args: argparse.Namespace, log_folder: str, train_batch: EveryDreamBatch, epoch: int):
    if args.write_schedule:
        with open(f"{log_folder}/ep{epoch}_batch_schedule.txt", "w", encoding='utf-8') as f:
            for i in range(len(train_batch.image_train_items)):
                try:
                    item = train_batch.image_train_items[i]
                    f.write(f"step:{int(i / train_batch.batch_size):05}, wh:{item.target_wh}, r:{item.runt_size}, path:{item.pathname}\n")
                except Exception as e:
                    logging.error(f" * Error writing to batch schedule for file path: {item.pathname}")


def read_sample_prompts(sample_prompts_file_path: str):
    sample_prompts = []
    with open(sample_prompts_file_path, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())
    return sample_prompts

def log_args(log_writer, args):
    arglog = "args:\n"
    for arg, value in sorted(vars(args).items()):
        arglog += f"{arg}={value}, "
    log_writer.add_text("config", arglog)


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

    if args.seed == -1:
        args.seed = random.randint(0, 2**30)
    seed = args.seed
    logging.info(f" Seed: {seed}")
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpuid}")
        gpu = GPU(device)
        torch.backends.cudnn.benchmark = True
    else:
        logging.warning("*** Running on CPU. This is for testing loading/config parsing code only.")
        device = 'cpu'
        gpu = None

    log_folder = os.path.join(args.logdir, f"{args.project_name}_{log_time}")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    @torch.no_grad()
    def __save_model(save_path, unet, text_encoder, tokenizer, scheduler, vae, optimizer, save_ckpt_dir, yaml_name, save_full_precision=False, save_optimizer_flag=False):
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
            save_ckpt_dir = os.curdir

        half = not save_full_precision

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=save_path, checkpoint_path=sd_ckpt_full, half=half)

        if yaml_name and yaml_name != "v1-inference.yaml":
            yaml_save_path = f"{os.path.join(save_ckpt_dir, os.path.basename(save_path))}.yaml"
            logging.info(f" * Saving yaml to {yaml_save_path}")
            shutil.copyfile(yaml_name, yaml_save_path)


        if save_optimizer_flag:
            optimizer_path = os.path.join(save_path, "optimizer.pt")
            logging.info(f" Saving optimizer state to {save_path}")
            save_optimizer(optimizer, optimizer_path)

    try:

        # check for a local file
        hf_cache_path = get_hf_ckpt_cache_path(args.resume_ckpt)
        if os.path.exists(hf_cache_path) or os.path.exists(args.resume_ckpt):
            model_root_folder, is_sd1attn, yaml = convert_to_hf(args.resume_ckpt)
            text_encoder = CLIPTextModel.from_pretrained(model_root_folder, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model_root_folder, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(model_root_folder, subfolder="unet")

            optimizer_state_path = os.path.join(args.resume_ckpt, "optimizer.pt")
            if not os.path.exists(optimizer_state_path):
                optimizer_state_path = None
        else:
            # try to download from HF using resume_ckpt as a repo id
            downloaded = try_download_model_from_hf(repo_id=args.resume_ckpt)
            if downloaded is None:
                raise ValueError(f"No local file/folder for {args.resume_ckpt}, and no matching huggingface.co repo could be downloaded")
            pipe, model_root_folder, is_sd1attn, yaml = downloaded
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            unet = pipe.unet
            del pipe

        reference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
        noise_scheduler = DDPMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(model_root_folder, subfolder="tokenizer", use_fast=False)

    except Exception as e:
        traceback.print_exc()
        logging.error(" * Failed to load checkpoint *")
        raise

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if not args.disable_xformers:
        if (args.amp and is_sd1attn) or (not is_sd1attn):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers")
            except Exception as ex:
                logging.warning("failed to load xformers, using attention slicing instead")
                unet.set_attention_slice("auto")
                pass
    else:
        logging.info("xformers disabled, using attention slicing instead")
        unet.set_attention_slice("auto")

    vae = vae.to(device, dtype=torch.float16 if args.amp else torch.float32)
    unet = unet.to(device, dtype=torch.float32)
    if args.disable_textenc_training and args.amp:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    else:
        text_encoder = text_encoder.to(device, dtype=torch.float32)

    optimizer_config = None
    optimizer_config_path  = args.optimizer_config if args.optimizer_config else "optimizer.json"
    if os.path.exists(os.path.join(os.curdir, optimizer_config_path)):
        with open(os.path.join(os.curdir, optimizer_config_path), "r") as f:
            optimizer_config = json.load(f)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir=log_folder, pytorch=False, tensorboard_x=False, save=False)
        wandb_run = wandb.init(
            project=args.project_name,
            config={"main_cfg": vars(args), "optimizer_cfg": optimizer_config},
            name=args.run_name,
            #sync_tensorboard=True, # broken?
            #dir=log_folder, # only for save, just duplicates the TB log to /{log_folder}/wandb ...
            )
        try:
            if webbrowser.get():
                webbrowser.open(wandb_run.url, new=2)
        except Exception:
            pass

    log_writer = SummaryWriter(log_dir=log_folder,
                               flush_secs=20,
                               comment=args.run_name if args.run_name is not None else log_time,
                              )

    betas = [0.9, 0.999]
    epsilon = 1e-8
    weight_decay = 0.01
    opt_class = None
    optimizer = None

    default_lr = 1e-6
    curr_lr = args.lr
    text_encoder_lr_scale = 1.0

    if optimizer_config is not None:
        betas = optimizer_config["betas"]
        epsilon = optimizer_config["epsilon"]
        weight_decay = optimizer_config["weight_decay"]
        optimizer_name = optimizer_config["optimizer"]
        curr_lr = optimizer_config.get("lr", curr_lr)
        if args.lr is not None:
            curr_lr = args.lr
            logging.info(f"Overriding LR from optimizer config with main config/cli LR setting: {curr_lr}")

        text_encoder_lr_scale = optimizer_config.get("text_encoder_lr_scale", text_encoder_lr_scale)
        if text_encoder_lr_scale != 1.0:
            logging.info(f" * Using text encoder LR scale {text_encoder_lr_scale}")

        logging.info(f" * Loaded optimizer args from {optimizer_config_path} *")

    if curr_lr is None:
        curr_lr = default_lr
        logging.warning(f"No LR setting found, defaulting to {default_lr}")

    curr_text_encoder_lr = curr_lr * text_encoder_lr_scale

    if args.disable_textenc_training:
        logging.info(f"{Fore.CYAN} * NOT Training Text Encoder, quality reduced *{Style.RESET_ALL}")
        params_to_train = itertools.chain(unet.parameters())
    elif args.disable_unet_training:
        logging.info(f"{Fore.CYAN} * Training Text Encoder Only *{Style.RESET_ALL}")
        if text_encoder_lr_scale != 1:
            logging.warning(f"{Fore.YELLOW} * Ignoring text_encoder_lr_scale {text_encoder_lr_scale} and using the "
                            f"Unet LR {curr_lr} for the text encoder instead *{Style.RESET_ALL}")
        params_to_train = itertools.chain(text_encoder.parameters())
    else:
        logging.info(f"{Fore.CYAN} * Training Text and Unet *{Style.RESET_ALL}")
        params_to_train = [{'params': unet.parameters()},
                           {'params': text_encoder.parameters(), 'lr': curr_text_encoder_lr}]

    if optimizer_name:
        if optimizer_name == "lion":
            from lion_pytorch import Lion
            opt_class = Lion
            optimizer = opt_class(
                itertools.chain(params_to_train),
                lr=curr_lr,
                betas=(betas[0], betas[1]),
                weight_decay=weight_decay,
            )
        elif optimizer_name in ["adamw"]:
            opt_class = torch.optim.AdamW
        else:
            import bitsandbytes as bnb
            opt_class = bnb.optim.AdamW8bit

    if not optimizer:
        optimizer = opt_class(
            itertools.chain(params_to_train),
            lr=curr_lr,
            betas=(betas[0], betas[1]),
            eps=epsilon,
            weight_decay=weight_decay,
            amsgrad=False,
        )

    if optimizer_state_path is not None:
        logging.info(f"Loading optimizer state from {optimizer_state_path}")
        load_optimizer(optimizer, optimizer_state_path)

    log_optimizer(optimizer, betas, epsilon, weight_decay, curr_lr, curr_text_encoder_lr)

    image_train_items = resolve_image_train_items(args, log_folder)

    validator = None
    if args.validation_config is not None:
        validator = EveryDreamValidator(args.validation_config,
                                        default_batch_size=args.batch_size,
                                        resolution=args.resolution,
                                        log_writer=log_writer,
                                        )
        # the validation dataset may need to steal some items from image_train_items
        image_train_items = validator.prepare_validation_splits(image_train_items, tokenizer=tokenizer)

    data_loader = DataLoaderMultiAspect(
        image_train_items=image_train_items,
        seed=seed,
        batch_size=args.batch_size,
    )

    train_batch = EveryDreamBatch(
        data_loader=data_loader,
        debug_level=1,
        conditional_dropout=args.cond_dropout,
        tokenizer=tokenizer,
        seed = seed,
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

    log_args(log_writer, args)

    sample_generator = SampleGenerator(log_folder=log_folder, log_writer=log_writer,
                                       default_resolution=args.resolution, default_seed=args.seed,
                                       config_file_path=args.sample_prompts,
                                       batch_size=max(1,args.batch_size//2),
                                       default_sample_steps=args.sample_steps,
                                       use_xformers=is_xformers_available() and not args.disable_xformers)

    """
    Train the model

    """
    print(f" {Fore.LIGHTGREEN_EX}** Welcome to EveryDream trainer 2.0!**{Style.RESET_ALL}")
    print(f" (C) 2022-2023 Victor C Hall  This program is licensed under AGPL 3.0 https://www.gnu.org/licenses/agpl-3.0.en.html")
    print()
    print("** Trainer Starting **")

    global interrupted
    interrupted = False

    def sigterm_handler(signum, frame):
        """
        handles sigterm
        """
        is_main_thread = (torch.utils.data.get_worker_info() == None)
        if is_main_thread:
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
                __save_model(interrupted_checkpoint_path, unet, text_encoder, tokenizer, noise_scheduler, vae, optimizer, args.save_ckpt_dir, args.save_full_precision, args.save_optimizer)
            exit(_SIGTERM_EXIT_CODE)
        else:
            # non-main threads (i.e. dataloader workers) should exit cleanly
            exit(0)

    signal.signal(signal.SIGINT, sigterm_handler)

    if not os.path.exists(f"{log_folder}/samples/"):
        os.makedirs(f"{log_folder}/samples/")

    if gpu is not None:
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        logging.info(f" Pretraining GPU Memory: {gpu_used_mem} / {gpu_total_mem} MB")
    logging.info(f" saving ckpts every {args.ckpt_every_n_minutes} minutes")
    logging.info(f" saving ckpts every {args.save_every_n_epochs } epochs")

    train_dataloader = build_torch_dataloader(train_batch, batch_size=args.batch_size)

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

    scaler = GradScaler(
        enabled=args.amp,
        init_scale=2**17.5,
        growth_factor=2,
        backoff_factor=1.0/2,
        growth_interval=25,
    )
    logging.info(f" Grad scaler enabled: {scaler.is_enabled()} (amp mode)")

    epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True, dynamic_ncols=True)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")
    epoch_times = []

    global global_step
    global_step = 0
    training_start_time = time.time()
    last_epoch_saved_time = training_start_time

    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    loss_log_step = []

    assert len(train_batch) > 0, "train_batch is empty, check that your data_root is correct"

    # actual prediction function - shared between train and validate
    def get_model_prediction_and_target(image, tokens, zero_frequency_noise_ratio=0.0):
        with torch.no_grad():
            with autocast(enabled=args.amp):
                pixel_values = image.to(memory_format=torch.contiguous_format).to(unet.device)
                latents = vae.encode(pixel_values, return_dict=False)
            del pixel_values
            latents = latents[0].sample() * 0.18215

            if zero_frequency_noise_ratio > 0.0:
                # see https://www.crosslabs.org//blog/diffusion-with-offset-noise
                zero_frequency_noise = zero_frequency_noise_ratio * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
                noise = torch.randn_like(latents) + zero_frequency_noise
            else:
                noise = torch.randn_like(latents)

            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            cuda_caption = tokens.to(text_encoder.device)

        encoder_hidden_states = text_encoder(cuda_caption, output_hidden_states=True)

        if args.clip_skip > 0:
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(
                encoder_hidden_states.hidden_states[-args.clip_skip])
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

        return model_pred, target

    def generate_samples(global_step: int, batch):
        with isolate_rng():
            prev_sample_steps = sample_generator.sample_steps
            sample_generator.reload_config()
            if prev_sample_steps != sample_generator.sample_steps:
                next_sample_step = math.ceil((global_step + 1) / sample_generator.sample_steps) * sample_generator.sample_steps
                print(f" * SampleGenerator config changed, now generating images samples every " +
                      f"{sample_generator.sample_steps} training steps (next={next_sample_step})")
            sample_generator.update_random_captions(batch["captions"])
            inference_pipe = sample_generator.create_inference_pipe(unet=unet,
                                                                    text_encoder=text_encoder,
                                                                    tokenizer=tokenizer,
                                                                    vae=vae,
                                                                    diffusers_scheduler_config=reference_scheduler.config
                                                                    ).to(device)
            sample_generator.generate_samples(inference_pipe, global_step)

            del inference_pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Pre-train validation to establish a starting point on the loss graph
    if validator:
        validator.do_validation_if_appropriate(epoch=0, global_step=0,
                                               get_model_prediction_and_target_callable=get_model_prediction_and_target)

    # the sample generator might be configured to generate samples before step 0
    if sample_generator.generate_pretrain_samples:
        _, batch = next(enumerate(train_dataloader))
        generate_samples(global_step=0, batch=batch)

    try:
        write_batch_schedule(args, log_folder, train_batch, epoch = 0)

        for epoch in range(args.max_epochs):
            loss_epoch = []
            epoch_start_time = time.time()
            images_per_sec_log_step = []

            epoch_len = math.ceil(len(train_batch) / args.batch_size)
            steps_pbar = tqdm(range(epoch_len), position=1, leave=False, dynamic_ncols=True)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

            for step, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"], args.zero_frequency_noise_ratio)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                if batch["runt_size"] > 0:
                    loss_scale = batch["runt_size"] / args.batch_size
                    loss = loss * loss_scale

                scaler.scale(loss).backward()

                if args.clip_grad_norm is not None:
                    if not args.disable_unet_training:
                        torch.nn.utils.clip_grad_norm_(parameters=unet.parameters(), max_norm=args.clip_grad_norm)
                    if not args.disable_textenc_training:
                        torch.nn.utils.clip_grad_norm_(parameters=text_encoder.parameters(), max_norm=args.clip_grad_norm)

                if ((global_step + 1) % args.grad_accum == 0) or (step == epoch_len - 1):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step()

                loss_step = loss.detach().item()

                steps_pbar.set_postfix({"loss/step": loss_step}, {"gs": global_step})
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
                    if args.disable_textenc_training or args.disable_unet_training or text_encoder_lr_scale == 1:
                        log_writer.add_scalar(tag="hyperparamater/lr", scalar_value=curr_lr, global_step=global_step)
                    else:
                        log_writer.add_scalar(tag="hyperparamater/lr unet", scalar_value=curr_lr, global_step=global_step)
                        curr_text_encoder_lr = lr_scheduler.get_last_lr()[1]
                        log_writer.add_scalar(tag="hyperparamater/lr text encoder", scalar_value=curr_text_encoder_lr, global_step=global_step)
                    log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_local, global_step=global_step)
                    sum_img = sum(images_per_sec_log_step)
                    avg = sum_img / len(images_per_sec_log_step)
                    images_per_sec_log_step = []
                    if args.amp:
                        log_writer.add_scalar(tag="hyperparamater/grad scale", scalar_value=scaler.get_scale(), global_step=global_step)
                    log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)
                    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                    torch.cuda.empty_cache()

                if (global_step + 1) % sample_generator.sample_steps == 0:
                    generate_samples(global_step=global_step, batch=batch)

                min_since_last_ckpt =  (time.time() - last_epoch_saved_time) / 60

                if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                    last_epoch_saved_time = time.time()
                    logging.info(f"Saving model, {args.ckpt_every_n_minutes} mins at step {global_step}")
                    save_path = os.path.join(f"{log_folder}/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")
                    __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, optimizer, args.save_ckpt_dir, yaml, args.save_full_precision, args.save_optimizer)

                if epoch > 0 and epoch % args.save_every_n_epochs == 0 and step == 0 and epoch < args.max_epochs - 1 and epoch >= args.save_ckpts_from_n_epochs:
                    logging.info(f" Saving model, {args.save_every_n_epochs} epochs at step {global_step}")
                    save_path = os.path.join(f"{log_folder}/ckpts/{args.project_name}-ep{epoch:02}-gs{global_step:05}")
                    __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, optimizer, args.save_ckpt_dir, yaml, args.save_full_precision, args.save_optimizer)

                del batch
                global_step += 1
                update_grad_scaler(scaler, global_step, epoch, step) if args.amp else None
                # end of step

            steps_pbar.close()

            elapsed_epoch_time = (time.time() - epoch_start_time) / 60
            epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
            log_writer.add_scalar("performance/minutes per epoch", elapsed_epoch_time, global_step)

            epoch_pbar.update(1)
            if epoch < args.max_epochs - 1:
                train_batch.shuffle(epoch_n=epoch, max_epochs = args.max_epochs)
                write_batch_schedule(args, log_folder, train_batch, epoch + 1)

            loss_local = sum(loss_epoch) / len(loss_epoch)
            log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_local, global_step=global_step)

            if validator:
                validator.do_validation_if_appropriate(epoch+1, global_step, get_model_prediction_and_target)

            gc.collect()
            # end of epoch

        # end of training

        save_path = os.path.join(f"{log_folder}/ckpts/last-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, optimizer, args.save_ckpt_dir, yaml, args.save_full_precision, args.save_optimizer)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        save_path = os.path.join(f"{log_folder}/ckpts/errored-{args.project_name}-ep{epoch:02}-gs{global_step:05}")
        __save_model(save_path, unet, text_encoder, tokenizer, noise_scheduler, vae, optimizer, args.save_ckpt_dir, yaml, args.save_full_precision, args.save_optimizer)
        raise ex

    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")


if __name__ == "__main__":
    supported_resolutions = aspects.get_supported_resolutions()
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, argv = argparser.parse_known_args()

    if args.config is not None:
        print(f"Loading training config from {args.config}.")
        with open(args.config, 'rt') as f:
            args.__dict__.update(json.load(f))
            if len(argv) > 0:
                print(f"Config .json loaded but there are additional CLI arguments -- these will override values in {args.config}.")
    else:
        print("No config file specified, using command line args")

    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--amp", action="store_true",  default=False, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--disable_amp", action="store_true", default=False, help="disables training of text encoder (def: False)")
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
    argparser.add_argument("--optimizer_config", default="optimizer.json", help="Path to a JSON configuration file for the optimizer.  Default is 'optimizer.json'")
    argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
    argparser.add_argument("--resolution", type=int, default=512, help="resolution to train", choices=supported_resolutions)
    argparser.add_argument("--resume_ckpt", type=str, required=not ('resume_ckpt' in args), default="sd_v1-5_vae.ckpt", help="The checkpoint to resume from, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1 ")
    argparser.add_argument("--run_name", type=str, required=False, default=None, help="Run name for wandb (child of project name), and comment for tensorboard, (def: None)")
    argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt", help="Text file with prompts to generate test samples from, or JSON file with sample generator settings (default: sample_prompts.txt)")
    argparser.add_argument("--sample_steps", type=int, default=250, help="Number of steps between samples (def: 250)")
    argparser.add_argument("--save_ckpt_dir", type=str, default=None, help="folder to save checkpoints to (def: root training folder)")
    argparser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every n epochs, def: 0 (disabled)")
    argparser.add_argument("--save_ckpts_from_n_epochs", type=int, default=0, help="Only saves checkpoints starting an N epochs, def: 0 (disabled)")
    argparser.add_argument("--save_full_precision", action="store_true", default=False, help="save ckpts at full FP32")
    argparser.add_argument("--save_optimizer", action="store_true", default=False, help="saves optimizer state with ckpt, useful for resuming training later")
    argparser.add_argument("--scale_lr", action="store_true", default=False, help="automatically scale up learning rate based on batch size and grad accumulation (def: False)")
    argparser.add_argument("--seed", type=int, default=555, help="seed used for samples and shuffling, use -1 for random")
    argparser.add_argument("--shuffle_tags", action="store_true", default=False, help="randomly shuffles CSV tags in captions, for booru datasets")
    argparser.add_argument("--useadam8bit", action="store_true", default=False, help="deprecated, use --optimizer_config and optimizer.json instead")
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None, help="Path to a JSON configuration file for the validator.  Default is no validation.")
    argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02, help="adds zero frequency noise, for improving contrast (def: 0.0) use 0.0 to 0.15")

    # load CLI args to overwrite existing config args
    args = argparser.parse_args(args=argv, namespace=args)
    print(f" Args:")
    pprint.pprint(vars(args))
    main(args)
