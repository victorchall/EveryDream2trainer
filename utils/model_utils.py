import os
import logging
import shutil
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel
from utils.huggingface_downloader import try_download_model_from_hf
from utils.unet_utils import get_attn_yaml # Assuming this path is correct relative to utils directory

def get_hf_ckpt_cache_path(ckpt_path):
    return os.path.join("ckpt_cache", os.path.basename(ckpt_path))

def convert_to_hf(ckpt_path):
    hf_cache = get_hf_ckpt_cache_path(ckpt_path)
    # from utils.unet_utils import get_attn_yaml # Moved to top-level imports

    if os.path.isfile(ckpt_path):
        if not os.path.exists(hf_cache):
            os.makedirs(hf_cache)
            logging.info(f"Converting {ckpt_path} to Diffusers format")
            try:
                # Assuming convert script is accessible or refactored
                import utils.convert_original_stable_diffusion_to_diffusers as convert_script
                convert_script.convert(ckpt_path, hf_cache) # Corrected destination
            except Exception as e:
                logging.error(f"Failed to convert checkpoint {ckpt_path}: {e}")
                logging.info("Please manually convert the checkpoint to Diffusers format (one time setup), see readme.")
                # exit() # Avoid exiting in a utility function
                raise # Re-raise the exception so the caller can handle it
        else:
            logging.info(f"Found cached checkpoint at {hf_cache}")

        is_sd1attn, yaml_path = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml_path
    elif os.path.isdir(ckpt_path): # Changed from hf_cache to ckpt_path for diffusers models
        is_sd1attn, yaml_path = get_attn_yaml(ckpt_path)
        return ckpt_path, is_sd1attn, yaml_path
    else:
        # This case might indicate an issue or a direct model ID from HF
        # For a direct HF model ID, get_attn_yaml might not be applicable before download
        # logging.warning(f"Checkpoint path {ckpt_path} is not a file or directory. Attempting to treat as HF model ID.")
        # The original logic for this else block needs to be clarified or adapted,
        # as get_attn_yaml typically expects a local path.
        # For now, mirroring the original structure:
        is_sd1attn, yaml_path = get_attn_yaml(ckpt_path) # This line might fail if ckpt_path is not a local path
        return ckpt_path, is_sd1attn, yaml_path


def load_models(resume_ckpt: str, amp_enabled: bool, gradient_checkpointing: bool, attn_type: str, unet_in_bf16: bool = False):
    model_root_folder = None
    is_sd1attn = False
    yaml_path = None
    text_encoder = None
    vae = None
    unet = None

    # check for a local file
    hf_cache_path = get_hf_ckpt_cache_path(resume_ckpt)
    if os.path.exists(hf_cache_path) or os.path.exists(resume_ckpt):
        model_root_folder, is_sd1attn, yaml_path = convert_to_hf(resume_ckpt)
        text_encoder = CLIPTextModel.from_pretrained(model_root_folder, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model_root_folder, subfolder="vae")
        unet_dtype = torch.bfloat16 if unet_in_bf16 else torch.float32
        unet = UNet2DConditionModel.from_pretrained(model_root_folder, subfolder="unet", torch_dtype=unet_dtype)
    else:
        # try to download from HF using resume_ckpt as a repo id
        downloaded = try_download_model_from_hf(repo_id=resume_ckpt)
        if downloaded is None:
            raise ValueError(f"No local file/folder for {resume_ckpt}, and no matching huggingface.co repo could be downloaded")
        pipe, model_root_folder, is_sd1attn, yaml_path = downloaded
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet_dtype = torch.bfloat16 if unet_in_bf16 else torch.float32
        # Assuming pipe.unet can be loaded with a specific dtype directly, or handle conversion later
        unet = UNet2DConditionModel.from_pretrained(model_root_folder, subfolder="unet", torch_dtype=unet_dtype) # Re-load for dtype
        # unet = pipe.unet # This might not respect unet_in_bf16
        del pipe

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(text_encoder, 'gradient_checkpointing_enable'): # text_encoder might not always have this
            text_encoder.gradient_checkpointing_enable()

    if attn_type == "xformers":
        if (amp_enabled and is_sd1attn) or (not is_sd1attn):
            if is_xformers_available():
                try:
                    unet.enable_xformers_memory_efficient_attention()
                    logging.info("Enabled xformers")
                except Exception as ex:
                    logging.warning(f"failed to load xformers, using default SDP attention instead: {ex}")
            else:
                logging.warning("xformers is not available, using default SDP attention instead")
        elif (not amp_enabled and is_sd1attn):
            logging.info("AMP is disabled but model is SD1.X, xformers is incompatible so using default attention")
    elif attn_type == "slice":
        unet.set_attention_slice("auto")
    else:
        logging.info("* Using SDP attention *")
    
    return text_encoder, vae, unet, model_root_folder, is_sd1attn, yaml_path


def load_ema_models(ema_resume_model_path: str, device: torch.device):
    text_encoder_ema = None
    unet_ema = None

    logging.info(f"Loading EMA model: {ema_resume_model_path}")
    hf_cache_path = get_hf_ckpt_cache_path(ema_resume_model_path)

    if os.path.exists(hf_cache_path) or os.path.exists(ema_resume_model_path):
        ema_model_root_folder, _, _ = convert_to_hf(ema_resume_model_path) # is_sd1attn, yaml_path not needed for EMA here
        text_encoder_ema = CLIPTextModel.from_pretrained(ema_model_root_folder, subfolder="text_encoder")
        unet_ema = UNet2DConditionModel.from_pretrained(ema_model_root_folder, subfolder="unet")
    else:
        # try to download from HF using ema_resume_model_path as a repo id
        ema_downloaded = try_download_model_from_hf(repo_id=ema_resume_model_path)
        if ema_downloaded is None:
            raise ValueError(
                f"No local file/folder for ema_resume_model {ema_resume_model_path}, and no matching huggingface.co repo could be downloaded")
        ema_pipe, _, _, _ = ema_downloaded # ema_model_root_folder, ema_is_sd1attn, ema_yaml not strictly needed here after loading
        text_encoder_ema = ema_pipe.text_encoder
        unet_ema = ema_pipe.unet
        del ema_pipe

    # Move EMA models to the specified device
    if unet_ema is not None:
        unet_ema = unet_ema.to(device)
    if text_encoder_ema is not None:
        text_encoder_ema = text_encoder_ema.to(device)
            
    return text_encoder_ema, unet_ema

# It seems `release_memory` was mentioned in the context of EMA model loading in train.py
# If it's generally useful, it can be here. Otherwise, it might be specific to train.py's logic.
# For now, I'll include a basic version if it's needed by the moved logic.
import gc
def release_memory(model_to_delete, original_device_str: str): # device as string e.g. "cuda:0"
    del model_to_delete
    gc.collect()
    if 'cuda' in original_device_str: # Check if original_device was a CUDA device
        torch.cuda.empty_cache()

def get_training_noise_scheduler(train_sampler: str, model_root_folder, trained_betas=None):
    from diffusers import PNDMScheduler, DDIMScheduler, DDPMScheduler # Local import for clarity
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

def get_inference_scheduler(model_root_folder: str, enable_zero_terminal_snr: bool):
    from diffusers import DDIMScheduler # Local import
    from utils.unet_utils import enforce_zero_terminal_snr # Assuming this path is correct

    if enable_zero_terminal_snr:
        # Use zero terminal SNR
        temp_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
        trained_betas = enforce_zero_terminal_snr(temp_scheduler.betas).numpy().tolist()
        inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=trained_betas)
    else:
        inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
    return inference_scheduler

def load_tokenizer(model_root_folder: str):
    from transformers import CLIPTokenizer # Local import
    tokenizer = CLIPTokenizer.from_pretrained(model_root_folder, subfolder="tokenizer", use_fast=False)
    return tokenizer

def setup_model_for_training(unet, text_encoder, vae, amp_enabled, device_str, disable_textenc_training, disable_unet_training=False, gradient_checkpointing=False): # Added disable_unet_training and gradient_checkpointing
    # VAE to device and dtype
    vae_dtype = torch.float16 if amp_enabled else torch.float32
    vae = vae.to(torch.device(device_str), dtype=vae_dtype)

    # UNet to device and dtype (always float32 for training initially, can be changed by accelerator or specific settings like unet_in_bf16)
    unet = unet.to(torch.device(device_str), dtype=torch.float32)

    # TextEncoder to device and dtype
    if disable_textenc_training and amp_enabled:
        text_encoder_dtype = torch.float16
    else:
        text_encoder_dtype = torch.float32
    text_encoder = text_encoder.to(torch.device(device_str), dtype=text_encoder_dtype)
    
    # Gradient checkpointing (already handled in load_models, but can be double-checked or applied here if models are reloaded/modified)
    if gradient_checkpointing:
        if not disable_unet_training : # Only if UNet is being trained
            unet.enable_gradient_checkpointing()
        if not disable_textenc_training and hasattr(text_encoder, 'gradient_checkpointing_enable'):
             text_encoder.gradient_checkpointing_enable()

    # Set train/eval modes
    if not disable_unet_training or gradient_checkpointing: # unet.train() if gradient_checkpointing is true, even if unet training disabled
        unet.train()
    else:
        unet.eval()

    if not disable_textenc_training:
        text_encoder.train()
    else:
        text_encoder.eval()
        
    return unet, text_encoder, vae

def create_ema_model_if_needed(use_ema_decay_training: bool, ema_model_loaded_from_file: bool, unet, text_encoder, ema_device_str: str, main_device_str: str):
    unet_ema = None
    text_encoder_ema = None
    if use_ema_decay_training:
        if not ema_model_loaded_from_file:
            logging.info(f"EMA decay enabled, creating EMA model.")
            with torch.no_grad():
                # Determine the target device for deepcopy
                target_device_for_copy = torch.device(main_device_str) # Always copy on the main device first
                
                unet_copy = deepcopy(unet.to(target_device_for_copy))
                text_encoder_copy = deepcopy(text_encoder.to(target_device_for_copy))

                # Move to ema_device if different
                ema_torch_device = torch.device(ema_device_str)
                if ema_torch_device != target_device_for_copy:
                    unet_ema = unet_copy.to(ema_torch_device)
                    text_encoder_ema = text_encoder_copy.to(ema_torch_device)
                    del unet_copy # free memory on main_device if copied then moved
                    del text_encoder_copy
                else:
                    unet_ema = unet_copy
                    text_encoder_ema = text_encoder_copy
        else:
            # This case implies ema models (unet_ema, text_encoder_ema) are already loaded
            # and just need to be ensured they are on the correct device and dtype.
            # This function's scope is creating them if *not* loaded.
            # If they are loaded, their management (device, dtype) should be handled
            # by the caller or another dedicated function.
            # For now, this branch will do nothing, assuming they are handled externally.
            logging.info("EMA models already loaded, skipping creation.")
            pass # unet_ema and text_encoder_ema should be passed in or handled by caller
    
    return unet_ema, text_encoder_ema

# Placeholder for convert_diff_to_ckpt.py's convert function if it's simple enough
# Or ensure it's properly importable if it's complex.
# from utils.convert_diff_to_ckpt import convert as convert_diffusers_to_sd_ckpt
import subprocess

def convert_diffusers_to_sd_ckpt(model_path, checkpoint_path, half=False):
    # This is a placeholder. The actual conversion might be more complex
    # and involve calling a script or specific library functions.
    # For example, using a command-line tool:
    try:
        command = [
            "python", "utils/convert_diff_to_ckpt.py", # Assuming script is in utils
            "--model_path", model_path,
            "--checkpoint_path", checkpoint_path
        ]
        if half:
            command.append("--half")
        
        logging.info(f"Executing conversion: {' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info(f"Successfully converted {model_path} to {checkpoint_path}")
    except FileNotFoundError:
        logging.error("Conversion script (utils/convert_diff_to_ckpt.py) not found. Please ensure it exists.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Conversion failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}")
        raise

# This is for the `save_model` function in `train.py`
# It's not directly requested to move `save_model` but `convert_diffusers_to_sd_ckpt` is part of it.
# For now, this is just a note.
# If `save_model` is moved, then `StableDiffusionPipeline` import would be needed here too.

# Added deepcopy which was missed in the original plan
from copy import deepcopy
