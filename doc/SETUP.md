## Install Python

Install Python 3.10 from here:

https://www.python.org/downloads/release/python-3109/

https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe

Download and install Git from [git-scm.com](https://git-scm.com/).

or [Git for windows](https://gitforwindows.org/)

## Clone this repo
Clone the repo from normal command line then change into the directory:

    git clone https://github.com/victorchall/EveryDream-trainer2

    cd EveryDream-trainer2

## Download models

You need some sort of base model to start training.  I suggest these two:

Stable Diffusion 1.5 with improved VAE:

https://huggingface.co/panopstor/EveryDream/blob/main/sd_v1-5_vae.ckpt

SD2.1 768:

https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-nonema-pruned.ckpt

You can use SD2.0 512 as well, but typically SD1.5 is going to be better.

https://huggingface.co/stabilityai/stable-diffusion-2-base/blob/main/512-base-ema.ckpt

Place these in the root folder of EveryDream2.

Run these commands *one time* to prepare them:

For SD1.x models, use this:

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v1-inference.yaml ^
    --image_size 768 ^
    --checkpoint_path sd_v1-5_vae.ckpt ^
    --prediction_type epsilon ^
    --upcast_attn False ^
    --pipeline_type FrozenOpenCLIPEmbedder ^
    --dump_path "ckpt_cache/sd_v1-5_vae"

And the SD2.1 768 model:

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v2-inference-v.yaml ^
    --image_size 768 ^
    --checkpoint_path v2-1_768-ema-pruned.ckpt ^
    --prediction_type v_prediction ^
    --upcast_attn False ^
    --pipeline_type FrozenOpenCLIPEmbedder ^
    --dump_path "ckpt_cache/v2-1_768-ema-pruned"

And finally the SD2.0 512 base model (generally not recommended base model):

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v2-inference.yaml ^
    --image_size 768 ^
    --checkpoint_path 512-base-ema.ckpt ^
    --prediction_type epsilon ^
    --upcast_attn False ^
    --pipeline_type FrozenOpenCLIPEmbedder ^
    --dump_path "ckpt_cache/512-base-ema"

If you have other models, you need to know the base model that was used for them, in particular use the correct yaml (original_config_file) or it will not properly convert.

All of the above is one time.  After running, you will use --resume_ckpt and just name the file without "ckpt_cache"

ex.

    python train.py --resume_ckpt "sd_v1-5_vae" ...
    python train.py --resume_ckpt "v2-1_768-ema-pruned" ...
    python train.py --resume_ckpt "512-base-ema" ...

## Windows


Run windows_setup.bat to create your venv and install dependencies.

    windows_setup.bat


## Linux, Linux containers, or WSL

TBD

