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
from typing import Optional

import torch.nn.functional as F
from torch.cuda.amp import autocast

from colorama import Fore, Style
import numpy as np
import itertools
import torch
import datetime
import json
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler, \
    DPMSolverMultistepScheduler, PNDMScheduler
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
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID
from plugins.plugins import PluginRunner
from utils.huggingface_downloader import try_download_model_from_hf
from utils.convert_diff_to_ckpt import convert as converter
from utils.isolate_rng import isolate_rng
from utils.check_git import check_git
from optimizer.optimizers import EveryDreamOptimizer
from copy import deepcopy

if torch.cuda.is_available():
    from utils.gpu import GPU
import data.aspects as aspects
import data.resolver as resolver
from utils.sample_generator import SampleGenerator

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

def get_training_noise_scheduler(train_sampler: str, model_root_folder, trained_betas=None):
    noise_scheduler = None
    if train_sampler.lower() == "pndm":
        logging.info(f" * Using PNDM noise scheduler for training: {train_sampler}")
        noise_scheduler = PNDMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
    elif train_sampler.lower() == "ddim":
        logging.info(f" * Using DDIM noise scheduler for training: {train_sampler}")
        noise_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
    else:
        logging.info(f" * Using default (DDPM) noise scheduler for training: {train_sampler}")
        noise_scheduler = DDPMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
    return noise_scheduler

def get_hf_ckpt_cache_path(ckpt_path):
    return os.path.join("ckpt_cache", os.path.basename(ckpt_path))

def convert_to_hf(ckpt_path):

    hf_cache = get_hf_ckpt_cache_path(ckpt_path)
    from utils.unet_utils import get_attn_yaml

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

class EveryDreamTrainingState:
    def __init__(self,
                 optimizer: Optional[EveryDreamOptimizer],
                 train_batch: Optional[EveryDreamBatch],
                 unet: UNet2DConditionModel,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 scheduler: Optional,
                 vae: AutoencoderKL,
                 unet_ema: Optional[UNet2DConditionModel],
                 text_encoder_ema: Optional[CLIPTextModel]
                 ):
        self.optimizer = optimizer
        self.train_batch = train_batch
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.vae = vae
        self.unet_ema = unet_ema
        self.text_encoder_ema = text_encoder_ema


@torch.no_grad()
def save_model(save_path, ed_state: EveryDreamTrainingState, global_step: int, save_ckpt_dir, yaml_name,
               save_full_precision=False, save_optimizer_flag=False, save_ckpt=True, plugin_runner: PluginRunner=None):
    """
    Save the model to disk
    """

    def save_ckpt_file(diffusers_model_path, sd_ckpt_path):
        nonlocal save_ckpt_dir
        nonlocal save_full_precision
        nonlocal yaml_name

        if save_ckpt_dir is not None:
            sd_ckpt_full = os.path.join(save_ckpt_dir, sd_ckpt_path)
        else:
            sd_ckpt_full = os.path.join(os.curdir, sd_ckpt_path)
            save_ckpt_dir = os.curdir

        half = not save_full_precision

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=diffusers_model_path, checkpoint_path=sd_ckpt_full, half=half)

        if yaml_name and yaml_name != "v1-inference.yaml":
            yaml_save_path = f"{os.path.join(save_ckpt_dir, os.path.basename(diffusers_model_path))}.yaml"
            logging.info(f" * Saving yaml to {yaml_save_path}")
            shutil.copyfile(yaml_name, yaml_save_path)

    if global_step is None or global_step == 0:
        logging.warning("  No model to save, something likely blew up on startup, not saving")
        return

    if ed_state.unet_ema is not None or ed_state.text_encoder_ema is not None:
        pipeline_ema = StableDiffusionPipeline(
            vae=ed_state.vae,
            text_encoder=ed_state.text_encoder_ema,
            tokenizer=ed_state.tokenizer,
            unet=ed_state.unet_ema,
            scheduler=ed_state.scheduler,
            safety_checker=None, # save vram
            requires_safety_checker=None, # avoid nag
            feature_extractor=None, # must be none of no safety checker
        )

        diffusers_model_path = save_path + "_ema"
        logging.info(f" * Saving diffusers EMA model to {diffusers_model_path}")
        pipeline_ema.save_pretrained(diffusers_model_path)

        if save_ckpt:
            sd_ckpt_path_ema = f"{os.path.basename(save_path)}_ema.safetensors"

            save_ckpt_file(diffusers_model_path, sd_ckpt_path_ema)


    pipeline = StableDiffusionPipeline(
        vae=ed_state.vae,
        text_encoder=ed_state.text_encoder,
        tokenizer=ed_state.tokenizer,
        unet=ed_state.unet,
        scheduler=ed_state.scheduler,
        safety_checker=None,  # save vram
        requires_safety_checker=None,  # avoid nag
        feature_extractor=None,  # must be none of no safety checker
    )

    diffusers_model_path = save_path
    logging.info(f" * Saving diffusers model to {diffusers_model_path}")
    pipeline.save_pretrained(diffusers_model_path)

    if save_ckpt:
        sd_ckpt_path = f"{os.path.basename(save_path)}.safetensors"
        save_ckpt_file(diffusers_model_path, sd_ckpt_path)

    if save_optimizer_flag:
        logging.info(f" Saving optimizer state to {save_path}")
        ed_state.optimizer.save(save_path)

    plugin_runner.run_on_model_save(
        ed_state=ed_state,
        diffusers_save_path=diffusers_model_path
    )


def setup_local_logger(args):
    """
    configures logger with file and console logging, logs args, and returns the datestamp
    """
    log_path = args.logdir
    os.makedirs(log_path, exist_ok=True)
    
    datetimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_folder = os.path.join(log_path, f"{args.project_name}-{datetimestamp}")
    os.makedirs(log_folder, exist_ok=True)

    logfilename = os.path.join(log_folder, f"{args.project_name}-{datetimestamp}.log")

    print(f" logging to {logfilename}")
    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                       )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(lambda msg: "Palette images with Transparency expressed in bytes" not in msg.getMessage())
    logging.getLogger().addHandler(console_handler)
    import warnings
    warnings.filterwarnings("ignore", message="UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images")
    #from PIL import Image

    return datetimestamp, log_folder

# def save_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Saves the optimizer state
#     """
#     torch.save(optimizer.state_dict(), path)

# def load_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Loads the optimizer state
#     """
#     optimizer.load_state_dict(torch.load(path))

def pyramid_noise_like(x, discount=0.8):
  b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
  u = torch.nn.Upsample(size=(w, h), mode='bilinear')
  noise = torch.randn_like(x)
  for i in range(10):
    r = random.random()*2+2 # Rather than always going 2x, 
    w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
    noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
    if w==1 or h==1: break # Lowest resolution is 1x1
  return noise/noise.std() # Scaled back to roughly unit variance

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

def find_last_checkpoint(logdir, is_ema=False):
    """
    Finds the last checkpoint in the logdir, recursively
    """
    last_ckpt = None
    last_date = None

    for root, dirs, files in os.walk(logdir):
        for file in files:
            if os.path.basename(file) == "model_index.json":

                if is_ema and (not root.endswith("_ema")):
                    continue
                elif (not is_ema) and root.endswith("_ema"):
                    continue

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

    if (args.ema_resume_model != None) and (args.ema_resume_model == "findlast"):
        logging.info(f"{Fore.LIGHTCYAN_EX} Finding last EMA decay checkpoint in logdir: {args.logdir}{Style.RESET_ALL}")

        args.ema_resume_model = find_last_checkpoint(args.logdir, is_ema=True)

    if not args.shuffle_tags:
        args.shuffle_tags = False

    if not args.keep_tags:
        args.keep_tags = 0

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


    if args.save_ckpt_dir is not None and not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    if args.rated_dataset:
        args.rated_dataset_target_dropout_percent = min(max(args.rated_dataset_target_dropout_percent, 0), 100)

        logging.info(logging.info(f"{Fore.CYAN} * Activating rated images learning with a target rate of {args.rated_dataset_target_dropout_percent}% {Style.RESET_ALL}"))

    args.aspects = aspects.get_aspect_buckets(args.resolution)

    return args


def report_image_train_item_problems(log_folder: str, items: list[ImageTrainItem], batch_size) -> None:
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


    # warn on underfilled aspect ratio buckets

    # Intuition: if there are too few images to fill a batch, duplicates will be appended.
    # this is not a problem for large image counts but can seriously distort training if there
    # are just a handful of images for a given aspect ratio.

    # at a dupe ratio of 0.5, all images in this bucket have effective multiplier 1.5,
    # at a dupe ratio 1.0, all images in this bucket have effective multiplier 2.0
    warn_bucket_dupe_ratio = 0.5

    def make_bucket_key(item):
        return (item.batch_id, int(item.target_wh[0]), int(item.target_wh[1]))

    ar_buckets = set(make_bucket_key(i) for i in items)
    for ar_bucket in ar_buckets:
        count = len([i for i in items if make_bucket_key(i) == ar_bucket])
        runt_size = batch_size - (count % batch_size)
        bucket_dupe_ratio = runt_size / count
        if bucket_dupe_ratio > warn_bucket_dupe_ratio:
            aspect_ratio_rational = aspects.get_rational_aspect_ratio((ar_bucket[1], ar_bucket[2]))
            aspect_ratio_description = f"{aspect_ratio_rational[0]}:{aspect_ratio_rational[1]}"
            batch_id_description = "" if ar_bucket[0] == DEFAULT_BATCH_ID else f" for batch id '{ar_bucket[0]}'"
            effective_multiplier = round(1 + bucket_dupe_ratio, 1)
            logging.warning(f" * {Fore.LIGHTRED_EX}Aspect ratio bucket {ar_bucket} has only {count} "
                            f"images{Style.RESET_ALL}. At batch size {batch_size} this makes for an effective multiplier "
                            f"of {effective_multiplier}, which may cause problems. Consider adding {runt_size} or "
                            f"more images with aspect ratio {aspect_ratio_description}{batch_id_description}, or reducing your batch_size.")

def resolve_image_train_items(args: argparse.Namespace) -> list[ImageTrainItem]:
    logging.info(f"* DLMA resolution {args.resolution}, buckets: {args.aspects}")
    logging.info(" Preloading images...")

    resolved_items = resolver.resolve(args.data_root, args)
    image_paths = set(map(lambda item: item.pathname, resolved_items))

    # Remove erroneous items
    for item in resolved_items:
        if item.error is not None:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {item.error}")
    image_train_items = [item for item in resolved_items if item.error is None]
    print (f" * Found {len(image_paths)} files in '{args.data_root}'")

    return image_train_items

def write_batch_schedule(log_folder: str, train_batch: EveryDreamBatch, epoch: int):
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


def log_args(log_writer, args, optimizer_config, log_folder, log_time):
    arglog = "args:\n"
    for arg, value in sorted(vars(args).items()):
        arglog += f"{arg}={value}, "
    log_writer.add_text("config", arglog)

    args_as_json = json.dumps(vars(args), indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_main.json"), "w") as f:
        f.write(args_as_json)
        
    optimizer_config_as_json = json.dumps(optimizer_config, indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_opt.json"), "w") as f:
        f.write(optimizer_config_as_json)


def update_ema(model, ema_model, decay, default_device, ema_device):
    with torch.no_grad():
        original_model_on_proper_device = model
        need_to_delete_original = False
        if ema_device != default_device:
            original_model_on_other_device = deepcopy(model)
            original_model_on_proper_device = original_model_on_other_device.to(ema_device, dtype=model.dtype)
            del original_model_on_other_device
            need_to_delete_original = True

        params = dict(original_model_on_proper_device.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name in ema_params:
            #ema_params[name].data.mul_(decay).add_(params[name].data, alpha=1 - decay)
            ema_params[name].data = ema_params[name] * decay + params[name].data * (1.0 - decay)

        if need_to_delete_original:
            del(original_model_on_proper_device)

def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    minimal_value = 1e-9
    alphas_cumprod = noise_scheduler.alphas_cumprod
    # Use .any() to check if any elements in the tensor are zero
    if (alphas_cumprod[:-1] == 0).any():
        logging.warning(
            f"Alphas cumprod has zero elements! Resetting to {minimal_value}.."
        )
        alphas_cumprod[alphas_cumprod[:-1] == 0] = minimal_value
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR, first without epsilon
    snr = (alpha / sigma) ** 2
    # Check if the first element in SNR tensor is zero
    if torch.any(snr == 0):
        snr[snr == 0] = minimal_value
    return snr

def load_train_json_from_file(args, report_load = False):
    try:
        if report_load:
            print(f"Loading training config from {args.config}.")

        with open(args.config, 'rt') as f:
            read_json = json.load(f)

        args.__dict__.update(read_json)
    except Exception as config_read:
        print(f"Error on loading training config from {args.config}.")

def main(args):
    """
    Main entry point
    """
    if os.name == 'nt':
        print(" * Windows detected, disabling Triton")
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"

    log_time, log_folder = setup_local_logger(args)
    args = setup_args(args)
    print(f" Args:")
    pprint.pprint(vars(args))

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

    #log_folder = os.path.join(args.logdir, f"{args.project_name}_{log_time}")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    def release_memory(model_to_delete, original_device):
        del model_to_delete
        gc.collect()

        if 'cuda' in original_device.type:
            torch.cuda.empty_cache()


    use_ema_dacay_training = (args.ema_decay_rate != None) or (args.ema_strength_target != None)
    ema_model_loaded_from_file = False

    if use_ema_dacay_training:
        ema_device = torch.device(args.ema_device)

    optimizer_state_path = None

    try:
        # check for a local file
        hf_cache_path = get_hf_ckpt_cache_path(args.resume_ckpt)
        if os.path.exists(hf_cache_path) or os.path.exists(args.resume_ckpt):
            model_root_folder, is_sd1attn, yaml = convert_to_hf(args.resume_ckpt)
            text_encoder = CLIPTextModel.from_pretrained(model_root_folder, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model_root_folder, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(model_root_folder, subfolder="unet")
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

        if use_ema_dacay_training and args.ema_resume_model:
            print(f"Loading EMA model: {args.ema_resume_model}")
            ema_model_loaded_from_file=True
            hf_cache_path = get_hf_ckpt_cache_path(args.ema_resume_model)

            if os.path.exists(hf_cache_path) or os.path.exists(args.ema_resume_model):
                ema_model_root_folder, ema_is_sd1attn, ema_yaml = convert_to_hf(args.resume_ckpt)
                text_encoder_ema = CLIPTextModel.from_pretrained(ema_model_root_folder, subfolder="text_encoder")
                unet_ema = UNet2DConditionModel.from_pretrained(ema_model_root_folder, subfolder="unet")

            else:
                # try to download from HF using ema_resume_model as a repo id
                ema_downloaded = try_download_model_from_hf(repo_id=args.ema_resume_model)
                if ema_downloaded is None:
                    raise ValueError(
                        f"No local file/folder for ema_resume_model {args.ema_resume_model}, and no matching huggingface.co repo could be downloaded")
                ema_pipe, ema_model_root_folder, ema_is_sd1attn, ema_yaml = ema_downloaded
                text_encoder_ema = ema_pipe.text_encoder
                unet_ema = ema_pipe.unet
                del ema_pipe

            # Make sure EMA model is on proper device, and memory released if moved
            unet_ema_current_device = next(unet_ema.parameters()).device
            if ema_device != unet_ema_current_device:
                unet_ema_on_wrong_device = unet_ema
                unet_ema = unet_ema.to(ema_device)
                release_memory(unet_ema_on_wrong_device, unet_ema_current_device)

            # Make sure EMA model is on proper device, and memory released if moved
            text_encoder_ema_current_device = next(text_encoder_ema.parameters()).device
            if ema_device != text_encoder_ema_current_device:
                text_encoder_ema_on_wrong_device = text_encoder_ema
                text_encoder_ema = text_encoder_ema.to(ema_device)
                release_memory(text_encoder_ema_on_wrong_device, text_encoder_ema_current_device)


        if args.enable_zero_terminal_snr:
            # Use zero terminal SNR
            from utils.unet_utils import enforce_zero_terminal_snr
            temp_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
            trained_betas = enforce_zero_terminal_snr(temp_scheduler.betas).numpy().tolist()
            inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
            noise_scheduler = DDPMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
            noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder, trained_betas=trained_betas)
        else:
            inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
            noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder)

        tokenizer = CLIPTokenizer.from_pretrained(model_root_folder, subfolder="tokenizer", use_fast=False)

    except Exception as e:
        traceback.print_exc()
        logging.error(" * Failed to load checkpoint *")
        raise

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if args.attn_type == "xformers":
        if (args.amp and is_sd1attn) or (not is_sd1attn):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers")
            except Exception as ex:
                logging.warning("failed to load xformers, using default SDP attention instead")
                pass
        elif (args.disable_amp and is_sd1attn):
            logging.info("AMP is disabled but model is SD1.X, xformers is incompatible so using default attention")
    elif args.attn_type == "slice":
        unet.set_attention_slice("auto")
    else:
        logging.info("* Using SDP attention *")

    vae = vae.to(device, dtype=torch.float16 if args.amp else torch.float32)
    unet = unet.to(device, dtype=torch.float32)
    if args.disable_textenc_training and args.amp:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    else:
        text_encoder = text_encoder.to(device, dtype=torch.float32)


    if use_ema_dacay_training:
        if not ema_model_loaded_from_file:
            logging.info(f"EMA decay enabled, creating EMA model.")

            with torch.no_grad():
                if args.ema_device == device:
                    unet_ema = deepcopy(unet)
                    text_encoder_ema = deepcopy(text_encoder)
                else:
                    unet_ema_first = deepcopy(unet)
                    text_encoder_ema_first = deepcopy(text_encoder)
                    unet_ema = unet_ema_first.to(ema_device, dtype=unet.dtype)
                    text_encoder_ema = text_encoder_ema_first.to(ema_device, dtype=text_encoder.dtype)
                    del unet_ema_first
                    del text_encoder_ema_first
        else:
            # Make sure correct types are used for models
            unet_ema = unet_ema.to(ema_device, dtype=unet.dtype)
            text_encoder_ema = text_encoder_ema.to(ema_device, dtype=text_encoder.dtype)
    else:
        unet_ema = None
        text_encoder_ema = None

    try:
        #unet = torch.compile(unet)
        #text_encoder = torch.compile(text_encoder)
        #vae = torch.compile(vae)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        #logging.info("Successfully compiled models")
    except Exception as ex:
        logging.warning(f"Failed to compile model, continuing anyway, ex: {ex}")
        pass

    optimizer_config = None
    optimizer_config_path = args.optimizer_config if args.optimizer_config else "optimizer.json"
    if os.path.exists(os.path.join(os.curdir, optimizer_config_path)):
        with open(os.path.join(os.curdir, optimizer_config_path), "r") as f:
            optimizer_config = json.load(f)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir=log_folder, pytorch=False, tensorboard_x=False, save=False)
        wandb_run = wandb.init(
            project=args.project_name,
            config={"main_cfg": vars(args), "optimizer_cfg": optimizer_config},
            name=args.run_name,
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

    image_train_items = resolve_image_train_items(args)

    validator = None
    if args.validation_config is not None:
        validator = EveryDreamValidator(args.validation_config,
                                        default_batch_size=args.batch_size,
                                        resolution=args.resolution,
                                        log_writer=log_writer,
                                        )
        # the validation dataset may need to steal some items from image_train_items
        image_train_items = validator.prepare_validation_splits(image_train_items, tokenizer=tokenizer)

    report_image_train_item_problems(log_folder, image_train_items, batch_size=args.batch_size)

    from plugins.plugins import load_plugin
    if args.plugins is not None:
        plugins = [load_plugin(name) for name in args.plugins]
    else:
        logging.info("No plugins specified")
        plugins = []

    from plugins.plugins import PluginRunner
    plugin_runner = PluginRunner(plugins=plugins)
    plugin_runner.run_on_model_load(
        ed_state=EveryDreamTrainingState(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
                                         optimizer=None, train_batch=None, scheduler=noise_scheduler, unet_ema=None, text_encoder_ema=None),
        optimizer_config=optimizer_config,
        disable_unet_training=args.disable_unet_training,
        disable_textenc_training=args.disable_textenc_training
    )

    data_loader = DataLoaderMultiAspect(
        image_train_items=image_train_items,
        seed=seed,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum
    )

    train_batch = EveryDreamBatch(
        data_loader=data_loader,
        debug_level=1,
        conditional_dropout=args.cond_dropout,
        tokenizer=tokenizer,
        seed = seed,
        shuffle_tags=args.shuffle_tags,
        keep_tags=args.keep_tags,
        plugin_runner=plugin_runner,
        rated_dataset=args.rated_dataset,
        rated_dataset_dropout_target=(1.0 - (args.rated_dataset_target_dropout_percent / 100.0))
    )

    torch.cuda.benchmark = False

    epoch_len = math.ceil(len(train_batch) / args.batch_size)


    if use_ema_dacay_training:
        args.ema_update_interval = args.ema_update_interval * args.grad_accum
        if args.ema_strength_target != None:
            total_number_of_steps: float = epoch_len * args.max_epochs
            total_number_of_ema_update: float = total_number_of_steps / args.ema_update_interval
            args.ema_decay_rate = args.ema_strength_target ** (1 / total_number_of_ema_update)

            logging.info(f"ema_strength_target is {args.ema_strength_target}, calculated ema_decay_rate will be: {args.ema_decay_rate}.")

        logging.info(
            f"EMA decay enabled, with ema_decay_rate {args.ema_decay_rate}, ema_update_interval: {args.ema_update_interval}, ema_device: {args.ema_device}.")


    ed_optimizer = EveryDreamOptimizer(args,
                                       optimizer_config,
                                       text_encoder,
                                       unet,
                                       epoch_len,
                                       log_writer)

    log_args(log_writer, args, optimizer_config, log_folder, log_time)

    sample_generator = SampleGenerator(log_folder=log_folder, log_writer=log_writer,
                                       default_resolution=args.resolution, default_seed=args.seed,
                                       config_file_path=args.sample_prompts,
                                       batch_size=max(1,args.batch_size//2),
                                       default_sample_steps=args.sample_steps,
                                       use_xformers=args.attn_type == "xformers",
                                       use_penultimate_clip_layer=(args.clip_skip >= 2),
                                       guidance_rescale=0.7 if args.enable_zero_terminal_snr else 0
                                       )

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
                interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step}")
                print()
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                time.sleep(2) # give opportunity to ctrl-C again to cancel save
                save_model(interrupted_checkpoint_path, global_step=global_step, ed_state=make_current_ed_state(),
                           save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
                           save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, plugin_runner=plugin_runner)
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

    unet.train() if (args.gradient_checkpointing or not args.disable_unet_training) else unet.eval()
    text_encoder.train() if not args.disable_textenc_training else text_encoder.eval()

    logging.info(f" unet device: {unet.device}, precision: {unet.dtype}, training: {unet.training}")
    logging.info(f" text_encoder device: {text_encoder.device}, precision: {text_encoder.dtype}, training: {text_encoder.training}")
    logging.info(f" vae device: {vae.device}, precision: {vae.dtype}, training: {vae.training}")
    logging.info(f" scheduler: {noise_scheduler.__class__}")

    logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.grad_accum}{Style.RESET_ALL}"),
    logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")

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
    def get_model_prediction_and_target(image, tokens, zero_frequency_noise_ratio=0.0, return_loss=False, loss_scale=None):
        with torch.no_grad():
            with autocast(enabled=args.amp):
                pixel_values = image.to(memory_format=torch.contiguous_format).to(unet.device)
                latents = vae.encode(pixel_values, return_dict=False)
            del pixel_values
            latents = latents[0].sample() * 0.18215

            noise = torch.randn_like(latents)

            if args.pyramid_noise_discount != None:
                if 0 < args.pyramid_noise_discount:
                    noise = pyramid_noise_like(noise, discount=args.pyramid_noise_discount)

            if zero_frequency_noise_ratio != None:
                if zero_frequency_noise_ratio < 0:
                    zero_frequency_noise_ratio = 0

                # see https://www.crosslabs.org//blog/diffusion-with-offset-noise
                zero_frequency_noise = zero_frequency_noise_ratio * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
                noise = noise + zero_frequency_noise

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
            #print(f"types: {type(noisy_latents)} {type(timesteps)} {type(encoder_hidden_states)}")
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if return_loss:
            if loss_scale is None:
                loss_scale = torch.ones(model_pred.shape[0], dtype=torch.float)

            if args.min_snr_gamma is not None:
                snr = compute_snr(timesteps, noise_scheduler)

                mse_loss_weights = (
                        torch.stack(
                            [snr, args.min_snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                )
                mse_loss_weights[snr == 0] = 1.0
                loss_scale = loss_scale * mse_loss_weights.to(loss_scale.device)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_scale.to(unet.device)
            loss = loss.mean()

            return model_pred, target, loss

        else:
            return model_pred, target

    def generate_samples(global_step: int, batch):
        nonlocal unet
        nonlocal text_encoder
        nonlocal unet_ema
        nonlocal text_encoder_ema

        with isolate_rng():
            prev_sample_steps = sample_generator.sample_steps
            sample_generator.reload_config()
            if prev_sample_steps != sample_generator.sample_steps:
                next_sample_step = math.ceil((global_step + 1) / sample_generator.sample_steps) * sample_generator.sample_steps
                print(f" * SampleGenerator config changed, now generating images samples every " +
                      f"{sample_generator.sample_steps} training steps (next={next_sample_step})")
            sample_generator.update_random_captions(batch["captions"])

            models_info = []

            if (args.ema_decay_rate is None) or args.ema_sample_nonema_model:
                models_info.append({"is_ema": False, "swap_required": False})

            if (args.ema_decay_rate is not None) and args.ema_sample_ema_model:
                models_info.append({"is_ema": True, "swap_required": ema_device != device})

            for model_info in models_info:

                extra_info: str = ""

                if model_info["is_ema"]:
                    current_unet, current_text_encoder = unet_ema, text_encoder_ema
                    extra_info = "_ema"
                else:
                    current_unet, current_text_encoder = unet, text_encoder

                torch.cuda.empty_cache()


                if model_info["swap_required"]:
                    with torch.no_grad():
                        unet_unloaded = unet.to(ema_device)
                        del unet
                        text_encoder_unloaded = text_encoder.to(ema_device)
                        del text_encoder

                        current_unet = unet_ema.to(device)
                        del unet_ema
                        current_text_encoder = text_encoder_ema.to(device)
                        del text_encoder_ema
                        gc.collect()
                        torch.cuda.empty_cache()



                inference_pipe = sample_generator.create_inference_pipe(unet=current_unet,
                                                                        text_encoder=current_text_encoder,
                                                                        tokenizer=tokenizer,
                                                                        vae=vae,
                                                                        diffusers_scheduler_config=inference_scheduler.config
                                                                        ).to(device)
                sample_generator.generate_samples(inference_pipe, global_step, extra_info=extra_info, plugin_runner=plugin_runner)

                # Cleanup
                del inference_pipe

                if model_info["swap_required"]:
                    with torch.no_grad():
                        unet = unet_unloaded.to(device)
                        del unet_unloaded
                        text_encoder = text_encoder_unloaded.to(device)
                        del text_encoder_unloaded

                        unet_ema = current_unet.to(ema_device)
                        del current_unet
                        text_encoder_ema = current_text_encoder.to(ema_device)
                        del current_text_encoder

                gc.collect()
                torch.cuda.empty_cache()

    def make_save_path(epoch, global_step, prepend=""):
        basename = f"{prepend}{args.project_name}"
        if epoch is not None:
            basename += f"-ep{epoch:02}"
        if global_step is not None:
            basename += f"-gs{global_step:05}"
        return os.path.join(log_folder, "ckpts", basename)


    # Pre-train validation to establish a starting point on the loss graph
    if validator:
        validator.do_validation(global_step=0,
                                get_model_prediction_and_target_callable=get_model_prediction_and_target)

    # the sample generator might be configured to generate samples before step 0
    if sample_generator.generate_pretrain_samples:
        _, batch = next(enumerate(train_dataloader))
        generate_samples(global_step=0, batch=batch)

    def make_current_ed_state() -> EveryDreamTrainingState:
        return EveryDreamTrainingState(optimizer=ed_optimizer,
                                       train_batch=train_batch,
                                       unet=unet,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       scheduler=noise_scheduler,
                                       vae=vae,
                                       unet_ema=unet_ema,
                                       text_encoder_ema=text_encoder_ema)

    epoch = None
    try:        
        plugin_runner.run_on_training_start(log_folder=log_folder, project_name=args.project_name)

        for epoch in range(args.max_epochs):
            write_batch_schedule(log_folder, train_batch, epoch) if args.write_schedule else None
            if args.load_settings_every_epoch:
                load_train_json_from_file(args)

            epoch_len = math.ceil(len(train_batch) / args.batch_size)

            def update_arg(arg: str, newValue):
                if arg == "grad_accum":
                    args.grad_accum = newValue
                    data_loader.grad_accum = newValue
                else:
                    raise("Unrecognized arg: " + arg)

            plugin_runner.run_on_epoch_start(
                epoch=epoch,
                global_step=global_step,
                epoch_length=epoch_len,
                project_name=args.project_name,
                log_folder=log_folder,
                data_root=args.data_root,
                arg_update_callback=update_arg
            )


            loss_epoch = []
            epoch_start_time = time.time()
            images_per_sec_log_step = []

            steps_pbar = tqdm(range(epoch_len), position=1, leave=False, dynamic_ncols=True)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

            validation_steps = (
                [] if validator is None
                else validator.get_validation_step_indices(epoch, len(train_dataloader))
            )

            for step, batch in enumerate(train_dataloader):

                step_start_time = time.time()

                plugin_runner.run_on_step_start(epoch=epoch,
                        local_step=step,
                        global_step=global_step,
                        project_name=args.project_name,
                        log_folder=log_folder,
                        batch=batch,
                        ed_state=make_current_ed_state())

                model_pred, target, loss = get_model_prediction_and_target(batch["image"],
                                                                           batch["tokens"],
                                                                           args.zero_frequency_noise_ratio,
                                                                           return_loss=True,
                                                                           loss_scale=batch["loss_scale"])

                del target, model_pred

                if batch["runt_size"] > 0:
                    runt_loss_scale = (batch["runt_size"] / args.batch_size)**1.5 # further discount runts by **1.5
                    loss = loss * runt_loss_scale

                ed_optimizer.step(loss, step, global_step, plugin_runner=plugin_runner, ed_state=make_current_ed_state())

                if args.ema_decay_rate != None:
                    if ((global_step + 1) % args.ema_update_interval) == 0:
                        # debug_start_time = time.time() # Measure time

                        if args.disable_unet_training != True:
                            update_ema(unet, unet_ema, args.ema_decay_rate, default_device=device, ema_device=ema_device)

                        if args.disable_textenc_training != True:
                            update_ema(text_encoder, text_encoder_ema, args.ema_decay_rate, default_device=device, ema_device=ema_device)

                        # debug_end_time = time.time() # Measure time
                        # debug_elapsed_time = debug_end_time - debug_start_time # Measure time
                        # print(f"Command update_EMA unet and TE took {debug_elapsed_time:.3f} seconds.") # Measure time


                loss_step = loss.detach().item()

                steps_pbar.set_postfix({"loss/step": loss_step}, {"gs": global_step})
                steps_pbar.update(1)

                images_per_sec = args.batch_size / (time.time() - step_start_time)
                images_per_sec_log_step.append(images_per_sec)

                loss_log_step.append(loss_step)
                loss_epoch.append(loss_step)

                if (global_step + 1) % args.log_step == 0:
                    loss_step = sum(loss_log_step) / len(loss_log_step)
                    lr_unet = ed_optimizer.get_unet_lr()
                    lr_textenc = ed_optimizer.get_textenc_lr()
                    loss_log_step = []

                    log_writer.add_scalar(tag="hyperparameter/lr unet", scalar_value=lr_unet, global_step=global_step)
                    log_writer.add_scalar(tag="hyperparameter/lr text encoder", scalar_value=lr_textenc, global_step=global_step)
                    log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_step, global_step=global_step)

                    sum_img = sum(images_per_sec_log_step)
                    avg = sum_img / len(images_per_sec_log_step)
                    images_per_sec_log_step = []
                    if args.amp:
                        log_writer.add_scalar(tag="hyperparameter/grad scale", scalar_value=ed_optimizer.get_scale(), global_step=global_step)
                    log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)

                    logs = {"loss/log_step": loss_step, "lr_unet": lr_unet, "lr_te": lr_textenc, "img/s": images_per_sec}
                    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                    torch.cuda.empty_cache()

                if validator and step in validation_steps:
                    validator.do_validation(global_step, get_model_prediction_and_target)

                if (global_step + 1) % sample_generator.sample_steps == 0:
                    generate_samples(global_step=global_step, batch=batch)

                min_since_last_ckpt =  (time.time() - last_epoch_saved_time) / 60

                needs_save = False
                if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                    last_epoch_saved_time = time.time()
                    logging.info(f"Saving model, {args.ckpt_every_n_minutes} mins at step {global_step}")
                    needs_save = True
                if epoch > 0 and epoch % args.save_every_n_epochs == 0 and step == 0 and epoch < args.max_epochs and epoch >= args.save_ckpts_from_n_epochs:
                    logging.info(f" Saving model, {args.save_every_n_epochs} epochs at step {global_step}")
                    needs_save = True
                if needs_save:
                    save_path = make_save_path(epoch, global_step)
                    save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
                               save_ckpt_dir=args.save_ckpt_dir, yaml_name=None,
                               save_full_precision=args.save_full_precision,
                               save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, plugin_runner=plugin_runner)

                plugin_runner.run_on_step_end(epoch=epoch,
                                      global_step=global_step,
                                      local_step=step,
                                      project_name=args.project_name,
                                      log_folder=log_folder,
                                      data_root=args.data_root,
                                      batch=batch,
                                      ed_state=make_current_ed_state())

                del batch
                global_step += 1
                # end of step

            steps_pbar.close()

            elapsed_epoch_time = (time.time() - epoch_start_time) / 60
            epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
            log_writer.add_scalar("performance/minutes per epoch", elapsed_epoch_time, global_step)

            plugin_runner.run_on_epoch_end(epoch=epoch,
                                           global_step=global_step,
                                           project_name=args.project_name,
                                           log_folder=log_folder,
                                           data_root=args.data_root,
                                           arg_update_callback=update_arg)

            epoch_pbar.update(1)
            if epoch < args.max_epochs - 1:
                train_batch.shuffle(epoch_n=epoch, max_epochs = args.max_epochs)

            if len(loss_epoch) > 0:
                loss_epoch = sum(loss_epoch) / len(loss_epoch)
                log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_epoch, global_step=global_step)

            gc.collect()
            # end of epoch

        # end of training
        epoch = args.max_epochs

        plugin_runner.run_on_training_end()

        save_path = make_save_path(epoch, global_step, prepend=("" if args.no_prepend_last else "last-"))
        save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
                   save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
                   save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, plugin_runner=plugin_runner)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        save_path = make_save_path(epoch, global_step, prepend="errored-")
        save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
                   save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
                   save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, plugin_runner=plugin_runner)
        logging.info(f"{Fore.LIGHTYELLOW_EX}Model saved, re-raising exception and exiting.  Exception was:{Style.RESET_ALL}{Fore.LIGHTRED_EX} {ex} {Style.RESET_ALL}")
        raise ex

    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")


if __name__ == "__main__":
    check_git()
    supported_resolutions = aspects.get_supported_resolutions()
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, argv = argparser.parse_known_args()

    load_train_json_from_file(args, report_load=True)

    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--amp", action="store_true",  default=True, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--attn_type", type=str, default="sdp", help="Attention mechanismto use", choices=["xformers", "sdp", "slice"])
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--disable_amp", action="store_true", default=False, help="disables automatic mixed precision (def: False)")
    argparser.add_argument("--disable_textenc_training", action="store_true", default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_unet_training", action="store_true", default=False, help="disables training of unet (def: False) NOT RECOMMENDED")
    argparser.add_argument("--flip_p", type=float, default=0.0, help="probability of flipping image horizontally (def: 0.0) use 0.0 to 1.0, ex 0.5, not good for specific faces!")
    argparser.add_argument("--gpuid", type=int, default=0, help="id of gpu to use for training, (def: 0) (ex: 1 to use GPU_ID 1), use nvidia-smi to find your GPU ids")
    argparser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="enable gradient checkpointing to reduce VRAM use, may reduce performance (def: False)")
    argparser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation factor (def: 1), (ex, 2)")
    argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
    argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default!")
    argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant")
    argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
    argparser.add_argument("--no_prepend_last", action="store_true", help="Do not prepend 'last-' to the final checkpoint filename")
    argparser.add_argument("--no_save_ckpt", action="store_true", help="Save only diffusers files, not .safetensors files (save disk space if you do not need LDM-style checkpoints)" )
    argparser.add_argument("--optimizer_config", default="optimizer.json", help="Path to a JSON configuration file for the optimizer.  Default is 'optimizer.json'")
    argparser.add_argument('--plugins', nargs='+', help='Names of plugins to use')
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
    argparser.add_argument("--seed", type=int, default=555, help="seed used for samples and shuffling, use -1 for random")
    argparser.add_argument("--shuffle_tags", action="store_true", default=False, help="randomly shuffles CSV tags in captions, for booru datasets")
    argparser.add_argument("--train_sampler", type=str, default="ddpm", help="noise sampler used for training, (default: ddpm)", choices=["ddpm", "pndm", "ddim"])
    argparser.add_argument("--keep_tags", type=int, default=0, help="Number of tags to keep when shuffle, used to randomly select subset of tags when shuffling is enabled, def: 0 (shuffle all)")
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None, help="Path to a JSON configuration file for the validator.  Default is no validation.")
    argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02, help="adds zero frequency noise, for improving contrast (def: 0.0) use 0.0 to 0.15")
    argparser.add_argument("--enable_zero_terminal_snr", action="store_true", default=None, help="Use zero terminal SNR noising beta schedule")
    argparser.add_argument("--load_settings_every_epoch", action="store_true", default=None, help="Enable reloading of 'train.json' at start of every epoch.")
    argparser.add_argument("--min_snr_gamma", type=int, default=None, help="min-SNR-gamma parameter is the loss function into individual tasks. Recommended values: 5, 1, 20. Disabled by default and enabled when used. More info: https://arxiv.org/abs/2303.09556")
    argparser.add_argument("--ema_decay_rate", type=float, default=None, help="EMA decay rate. EMA model will be updated with (1 - ema_rate) from training, and the ema_rate from previous EMA, every interval. Values less than 1 and not so far from 1. Using this parameter will enable the feature.")
    argparser.add_argument("--ema_strength_target", type=float, default=None, help="EMA decay target value in range (0,1). emarate will be calculated from equation: 'ema_decay_rate=ema_strength_target^(total_steps/ema_update_interval)'. Using this parameter will enable the ema feature and overide ema_decay_rate.")
    argparser.add_argument("--ema_update_interval", type=int, default=500, help="How many steps between optimizer steps that EMA decay updates. EMA model will be update on every step modulo grad_accum times ema_update_interval.")
    argparser.add_argument("--ema_device", type=str, default='cpu', help="EMA decay device values: cpu, cuda. Using 'cpu' is taking around 4 seconds per update vs fraction of a second on 'cuda'. Using 'cuda' will reserve around 3.2GB VRAM for a model, with 'cpu' the system RAM will be used.")
    argparser.add_argument("--ema_sample_nonema_model", action="store_true", default=False, help="Will show samples from non-EMA trained model, just like regular training. Can be used with: --ema_sample_ema_model")
    argparser.add_argument("--ema_sample_ema_model", action="store_true", default=False, help="Will show samples from EMA model. May be slower when using ema cpu offloading. Can be used with: --ema_sample_nonema_model")
    argparser.add_argument("--ema_resume_model", type=str, default=None, help="The EMA decay checkpoint to resume from for EMA decay, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1-ema-decay")
    argparser.add_argument("--pyramid_noise_discount", type=float, default=None, help="Enables pyramid noise and use specified discount factor for it")

    # load CLI args to overwrite existing config args
    args = argparser.parse_args(args=argv, namespace=args)

    main(args)
