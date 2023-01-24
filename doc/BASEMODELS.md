# Download and setup base models

In order to train, you need a base model on which to train.  This is a one-time setup to configure base models when you want to use a particular base.

Make sure the trainer is installed properly first. See [SETUP.md](SETUP.md) for more details. 

You can either [download one manually](#manual-download), or alternatively EveryDream2 can [automatically download](#automatic-download) a model from the Hugging Face hub for you.

## Manual download

First you have to download a `.ckpt` file for the base model, then you need to convert it to a "diffusers format" folder. When you finish you should see something like this, come back to reference this picture as you go through the steps below:

![models](ckptcache.png) *(this picture is just an EXAMPLE)*

### Downloading the .ckpt

I suggest one of these two models:

* Stable Diffusion 1.5 with improved VAE:

  https://huggingface.co/panopstor/EveryDream/blob/main/sd_v1-5_vae.ckpt


* SD2.1 768:

  https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-nonema-pruned.ckpt


* You can use SD2.0 512 as well, but typically SD1.5 is going to be better.
  https://huggingface.co/stabilityai/stable-diffusion-2-base/blob/main/512-base-ema.ckpt

Place these in the root folder of EveryDream2.

### Converting to 🧨diffusers format

Run these commands *one time* to prepare them. **It's very important to use the correct YAML!**

For SD1.x models, use this (note it will spill a lot of warnings to the console, but its fine):

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v1-inference.yaml ^
    --image_size 512 ^
    --checkpoint_path sd_v1-5_vae.ckpt ^
    --prediction_type epsilon ^
    --upcast_attn False ^
    --dump_path "ckpt_cache/sd_v1-5_vae"

And the SD2.1 768 model (uses v2-v yaml and "v_prediction" prediction type):

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v2-inference-v.yaml ^
    --image_size 768 ^
    --checkpoint_path v2-1_768-nonema-pruned.ckpt ^
    --prediction_type v_prediction ^
    --upcast_attn False ^
    --dump_path "ckpt_cache/v2-1_768-nonema-pruned"

And finally the SD2.0 512 base model (generally not recommended base model):

    python utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim ^
    --original_config_file v2-inference.yaml ^
    --image_size 512 ^
    --checkpoint_path 512-base-ema.ckpt ^
    --prediction_type epsilon ^
    --upcast_attn False ^
    --dump_path "ckpt_cache/512-base-ema"

If you have other models, you need to know the base model that was used for them, **in particular use the correct yaml (original_config_file) or it will not properly convert.** Make sure to put some sort of name in the dump_path after "ckpt_cache/" so you can reference it later.

All of the above is one time.  After running, you will use --resume_ckpt and just name the file without "ckpt_cache/"

ex.

    python train.py --resume_ckpt "sd_v1-5_vae" ...
    python train.py --resume_ckpt "v2-1_768-ema-pruned" ...
    python train.py --resume_ckpt "512-base-ema" ...

## Automatic download

If you don't want the hassle of downloading and converting ckpt files, you can pass a Hugging Face "repo id" for `--resume-ckpt` and the model will be automatically downloaded from Huggingface if it exists.

For example, to use [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1), you can pass the repo id `stabilityai/stable-diffusion-2-1` for `--resume_ckpt`:

    python train.py --resume_ckpt stabilityai/stable-diffusion-2-1 ...

You can use any model on Huggingface which is saved in 🧨diffusers format, which is the vast majority of models on [this list](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads). For example, to resume training using [nitrosocke's Modern Disney model](https://huggingface.co/nitrosocke/mo-di-diffusion) as a starting point, use `nitrosocke/mo-di-diffusion` as the repo id:

    python train.py --resume_ckpt nitrosocke/mo-di-diffusion ...

You can check if a 🧨diffusers format model is available by checking [the "Files" tab](https://huggingface.co/nitrosocke/mo-di-diffusion/tree/main) for a bunch of folders with names like `feature_extractor`, `unet`, and `vae`.

### Hugging Face login

If the model requires you to sign a license agreement, you may need to login to the Hugging Face hub before downloads will work. You can do this by running the following command in the terminal window before you start training:
   
    huggingface-cli login

When prompted, paste a [Hugging Face User Access Token](https://huggingface.co/settings/tokens) into the terminal window (you may not see anything appear to show that you've pasted something), and then press Enter.  

> To get an Access Token you'll need to [create a Hugging Face account](https://huggingface.co/join) if you don't have one already. Login to your account and click `New token` on [your User Access Tokens page](https://huggingface.co/settings/tokens) to create an Access Token that you can then copy and paste into the terminal window.

> Note that on Windows you may have to right-click the terminal window -> Paste, rather than just using ctrl-V. You also may not see anything appear in the terminal to indicate that you've pasted something - just press Enter anyway. If downloading doesn't work after setting a token, double-check you have agreed to the license agreement and try running `huggingface-cli login` again. 

**Alternatively**, you can set the environment variable `HF_API_TOKEN` to your Access Token. On Windows:

    set HF_API_TOKEN=<token>

On Linux:

    export HF_API_TOKEN=<token>

Replace `<token>` with the Access Token you got from [your Hugging Face User Access Tokens page](https://huggingface.co/settings/tokens)

### Where are the files?

By default the downloaded Hugging Face files are stored in the Hugging Face cache folder. On Windows this is at `C:\Users\username.cache\huggingface\hub`. On Linux it is at `~/.cache/huggingface/hub`. 

You can set the environment variable `HUGGINGFACE_HUB_CACHE` to change this. Eg, to put the cache on `Z:\stable-diffusion-big-files\huggingface-hub-cache` (on Windows):

    set HUGGINGFACE_HUB_CACHE=Z:\stable-diffusion-big-files\huggingface-hub-cache\

Make sure to do this before running `train.py`.

### Downloading from a subfolder

If the model you want to download is not stored in the root folder under the huggingface repo id, you can pass `--hf_repo_subfolder` to set the subfolder where it should be downloaded from.
