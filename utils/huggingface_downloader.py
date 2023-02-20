import logging
import os
from typing import Optional, Tuple

import huggingface_hub
from diffusers import StableDiffusionPipeline

from utils.analyze_unet import get_attn_yaml


def try_download_model_from_hf(repo_id: str) -> Tuple[StableDiffusionPipeline, str, bool, str] | None:
    """
    Attempts to download files from the following subfolders under the given repo id:
    "text_encoder", "vae", "unet", "scheduler", "tokenizer".
    :param repo_id The repository id of the model on huggingface, such as 'stabilityai/stable-diffusion-2-1' which
                    corresponds to `https://huggingface.co/stabilityai/stable-diffusion-2-1`.
    :param access_token Access token to use when fetching. If None, uses environment-saved token.
    :return: Root folder on disk to the downloaded files, or None if download failed.
    """

    access_token = os.environ.get('HF_API_TOKEN', None)
    if access_token is not None:
        huggingface_hub.login(access_token)

    # check if the model exists
    model_info = huggingface_hub.model_info(repo_id)
    if model_info is None:
        return None

    # load it to download it
    pipe, cache_folder = StableDiffusionPipeline.from_pretrained(repo_id, return_cached_folder=True)

    is_sd1_attn, yaml_path = get_attn_yaml(cache_folder)
    return pipe, cache_folder, is_sd1_attn, yaml_path
