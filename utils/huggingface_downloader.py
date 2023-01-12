import os
from typing import Optional

import huggingface_hub

def try_download_model_from_hf(repo_id: str,
                               subfolder: Optional[str]=None,
                               access_token: Optional[str]=None) -> Optional[str]:
    """
    Attempts to download files from the following subfolders under the given repo id:
    "text_encoder", "vae", "unet", "scheduler", "tokenizer".
    :param repo_id The repository id of the model on huggingface, such as 'stabilityai/stable-diffusion-2-1' which
                    corresponds to `https://huggingface.co/stabilityai/stable-diffusion-2-1`.
    :param access_token Access token to use when fetching. If None, uses environment-saved token.
    :return: Root folder on disk to the downloaded files, or None if download failed.
    """

    # login, if requested
    if access_token is not None:
        huggingface_hub.login(access_token)

    # check if the model exists
    model_info = huggingface_hub.model_info(repo_id)
    if model_info is None:
        return None

    model_subfolders = ["text_encoder", "vae", "unet", "scheduler", "tokenizer"]
    allow_patterns = [os.path.join(subfolder or '', f, "*") for f in model_subfolders]
    downloaded_folder = huggingface_hub.snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    return downloaded_folder
