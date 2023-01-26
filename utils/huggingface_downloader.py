import logging
import os
from typing import Optional, Tuple

import huggingface_hub
from utils.analyze_unet import get_attn_yaml


def try_download_model_from_hf(repo_id: str,
                               subfolder: Optional[str]=None) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """
    Attempts to download files from the following subfolders under the given repo id:
    "text_encoder", "vae", "unet", "scheduler", "tokenizer".
    :param repo_id The repository id of the model on huggingface, such as 'stabilityai/stable-diffusion-2-1' which
                    corresponds to `https://huggingface.co/stabilityai/stable-diffusion-2-1`.
    :param access_token Access token to use when fetching. If None, uses environment-saved token.
    :return: Root folder on disk to the downloaded files, or None if download failed.
    """

    try:
        access_token = os.environ['HF_API_TOKEN']
        if access_token is not None:
            huggingface_hub.login(access_token)
    except:
        logging.info("no HF_API_TOKEN env var found, will attempt to download without authenticating")

    # check if the model exists
    model_info = huggingface_hub.model_info(repo_id)
    if model_info is None:
        return None, None, None

    model_subfolders = ["text_encoder", "vae", "unet", "scheduler", "tokenizer"]
    allow_patterns = ["model_index.json"] + [os.path.join(subfolder or '', f, "*") for f in model_subfolders]
    # prefer *.bin files for now
    # TODO: look for *.safetensors files and download them instead, if they exist
    ignore_patterns = "*.safetensors"
    downloaded_folder = huggingface_hub.snapshot_download(repo_id=repo_id,
                                                          allow_patterns=allow_patterns,
                                                          ignore_patterns=ignore_patterns)
    print(f"model with repo id {repo_id} downloaded to {downloaded_folder}")
    is_sd1_attn, yaml_path = get_attn_yaml(downloaded_folder)
    return downloaded_folder, is_sd1_attn, yaml_path
