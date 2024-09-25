# Advanced Tweaking

This document is a bit more geared to experienced users who have trained several models.  It is not required reading for new users.

Start with the [Low VRAM guide](VRAM.md) if you are having trouble training on a 12GB card.

## Resolution

You can train resolutions from 512 to 1024 in 64 pixel increments.  General results from the community indicate you can push the base model a bit beyond what it was designed for *with enough training*.  This will work out better when you have a lot of training data (hundreds+) and enable slightly higher resolution at inference time without seeing repeats in your generated images.  This does cost speed of training and higher VRAM use!  Ex. 768 takes a significant amount of additional VRAM than 512, so you will need to compensate for that by reducing ```batch_size```.

    --resolution 640 ^

For instance, if training from the base 1.5 model, you can try trying at 576, 640, or 704.

If you are training on a base model that is 768, such as "SD 2.1 768-v", you should also probably use 768 as a base number and adjust from there.

Some results from the community seem to indicate training at a higher resolution on SD1.x models may increase how fast the model learns, and it may be a good idea to slightly reduce your learning rate as you increase resolution.  My suspicion is that the higher resolutions increase the gradients as more information is presented to the model per image.  

 You may need to experiment with the LR as you increase resolution. I don't have a perfect rule of thumb here, but I might suggest if you train SD1.5 which is a 512 model at resolution 768 you reduce your LR by about half.  ED2 tends to prefer ~2e-6 to ~5e-6 for normal 512 training on SD1.X models around batch 6-8, so if you train SD1.X at 768 consider 1e-6 to 2.5e-6 instead.  

## Log and ckpt save folders

If you want to use a nondefault location for saving logs or ckpt files, these:

Logdir defaults to the "logs" folder in the trainer directory.  If you want to save all logs (including diffuser copies of ckpts, sample images, and tensbooard events) use this:

    --logdir "/workspace/mylogs"

Remember to use the same folder when you launch tensorboard (```tensorboard --logdir "/worksapce/mylogs"```) or it won't find your logs.

By default the CKPT format copies of ckpts that are periodically saved are saved in the trainer root folder.  If you want to save them elsewhere, use this:

    --save_ckpt_dir "r:\webui\models\stable-diffusion"

This is useful if you want to dump the CKPT files directly to your webui/inference program model folder so you don't have to manually cut and paste it over.

## Conditional dropout

Conditional dropout means the prompt or caption on the training image is dropped, and the caption is "blank".  This can help with unconditional guidance, per the original paper and authors of Latent Diffusion and Stable Diffusion. This means the CFG Scale used at inference time will respond more smoothly. 

The value is defaulted at 0.04, which means 4% conditional dropout.  You can set it to 0.0 to disable it, or increase it.  For larger training (many tens of thousands) using 0.10 would be my recommendation.

This may also be useful to really "force" a style into the model with a high setting such as 0.15.  However, setting it very high may lead to bleeding or overfitting to your training data, especially if your data is not very diverse, which may or may not be desirable for your project.

    --cond_dropout 0.1 ^

## Conditional Embedding Perturbation

Paper: https://arxiv.org/pdf/2405.20494
    
    --embedding_perturbation 1.0

This is the gamma value in the paper.   This can be set to 0.0 to disable.  It adds gaussian noise to the embedding vector created by the text encoder.

The noise zero centered with a std_dev of (embedding_perturbation divided by the square root of the embedding dimension) of the text encoder (i.e. 768 for CLIP-L used in SD1.x).

$
\xi \backsim \mathcal{N} (0, \frac{\gamma}{\sqrt{\mathcal{d}}})
$

You can join the Discord server to see [experimental results](https://discord.com/channels/1026983422431862825/1247917538952740955).

## Timestep clamping

Stable Diffusion uses 1000 possible timesteps for denoising steps.  Timesteps are always chosen randomly per training example, per step, within the possible or allowed timesteps. 

If you wish to train only a portion of those timesteps instead of the entire schedule you can clamp the value.

For instance, if you only want to train from 500 to 999, use this:

    --timestep_start 500

Or if you only want to try from 0 to 449, use this:

    --timestep_end 450

Possible use cases are to "focus" training on aesthetics or composition by limiting timesteps and training specific data with certain qualities.  It's likely you may need to train all timesteps as a "clean up" if you train just specific timestep ranges first so the model does not overfit the fine tuned timesteps and lead to problems during inference.  

This could also be used to train expert models for specific timestep ranges, similar to the SDXL Refiner model. 

## Loss Type

You can change the type of loss from the standard [MSE ("L2") loss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) to [Huber loss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html), or a interpolated value across timesteps.  Valid values are "mse", "huber", "mse_huber", and "huber_mse".

    --loss_type huber

mse_huber will use MSE at timestep 0 and huber at timestep 999, and interpolate between the two across the intermediate timesteps. huber_mse is the reverse

[Experiment results](https://discord.com/channels/1026983422431862825/1229478214020235355) (Discord)

## LR tweaking

You should use [Optimizer config](doc/OPTIMZER.md) to tweak instead of the primary arg here, but it is left for legacy support of the Jupyter Notebook to make it easier to use the Jupyter Notbook in a happy path or simplified scenario.

    --lr 1.0e-6 ^
    
*If you set this in train.json or the main CLI arg it will override the value from your optimizer.json, so use with caution...*  Again, best to use optimizer.json instead.

## Clip skip

Clip skip counts back from the last hidden layer of the text encoder output for use as the text embedding.

*Note: since EveryDream2 uses HuggingFace Diffusers library, the penultimate layer is already selected when training and running inference on SD2.x models.*  This is defined in the text_encoder/config.json by the "num_hidden_layers" property of 23, which is penultimate out of the 24 layers and set by default in all diffusers SD2.x models.

    --clip_skip 2 ^

A value of "2" will count back one additional layer.  For SD1.x, "2" would be "penultimate" layer as commonly referred to in the community.  For SD2.x, it would be an *additional* layer back. 

*A value of "0" or "1" does nothing.*

### Cosine LR scheduler
Cosine LR scheduler will "taper off" your learning rate over time. It will reach a peak value of your ```--lr``` value then taper off following a cosine curve.  In other words, it allows you to set a high initial learning rate which lowers as training progresses.  This *may* help speed up training without overfitting.  If you wish to use this, I would set a slightly higher initial [learning rate](#lr-tweaking), maybe by 25-50% than you might use with a normal constant LR schedule.

Example:

    --lr_scheduler cosine ^

*I don't recommend trying to set the warmup and decay steps if you are using cosine, but they're here if you want them.*

There is also warmup with cosine secheuler, which will default to 2% of the decay steps.  You can manually set warmup, but it is typically more useful from training a brand new model from scratch, not for continuation training which we're all doing, thus the very short 2% default value. 

    --lr_warmup_steps 100 ^

Cosine scheduler also has a "decay period" to define how long it takes to get to zero LR as it tapers.  By default, the trainer sets this to slightly longer than it will take to get to your ```--max_epochs``` number of steps, so LR doesn't go all the way to zero and waste compute time near the end of training.   However, if you want to tweak, you have to set the number of steps yourself and estimate what that will be based on your max_epochs, batch_size, and number of training images.  **If you set this, be sure to watch your LR log in tensorboard to make sure it does what you expect.**

    --lr_decay_steps 2500 ^

If decay steps is too low, your LR will bottom out to zero, then start rising again, following a cosine waveform, which is probably a dumb idea.  If it is way too high, it will never taper off and you might as well use constant LR scheduler instead. 

## Gradient accumulation

Gradient accumulation is sort of like a virtual batch size increase, averaging the learning over more than one step (batch of images) before applying it to the model as an update to weights.

Example:

    --grad_accum 2 ^

The above example with combine the loss from 2 batches before applying updates.  This *may* be a good idea for higher resolution training that requires smaller batch size but mega batch sizes are also not the be-all-end all.

Some experimentation shows if you already have batch size in the 6-8 range than grad accumulation of more than 1 just reduces quality, but you can experiment. 

*There is some VRAM overhead to set grad_accum > 1*, about equal to increasing batch size by 1, but continuing to increase grad_accum to 3+ does not continue to increase VRAM use, while increasing batch size does.  This can still be useful if you are trying to train higher resolutions with a smaller batch size and gain the benefit of larger batch sizes in terms of generalization.  You will have to decide if this is worth using.  Currently it will not work on 12GB GPUs due to VRAM limitations.

## Gradient checkpointing

This is mostly useful to reduce VRAM for smaller GPUs, and together with AdamW 8 bit and AMP mode can enable <12GB GPU training.

Gradient checkpointing can also offer a higher batch size and/or higher resolution within whatever VRAM you have, so it may be useful even on a 24GB+ GPU if you specifically want to run a very large batch size.  The other option is using gradient accumulation instead.

    --gradient_checkpointing ^

While gradient checkpointing reduces performance, the ability to run a higher batch size brings performance back fairly close to without it. 

You may NOT want to use a batch size as large as 13-14, or you may find you need to tweak learning rate all over again to find the right balance.  Generally I would not turn it on for a 24GB GPU training at <640 resolution.

This probably IS a good idea for training at higher resolutions and allows >768 training on 24GB GPUs.  Balancing this toggle, resolution, and batch_size will take a few quick experiments to see what you can run safely.

## Flip_p

If you wish for your training images to be randomly flipped horizontally, use this to flip the images 50% of the time:

    --flip_p 0.5 ^

This is useful for styles or other training that is not asymmetrical.  It is not suggested for training specific human faces as it may wash out facial features as real people typically have at least some asymmetric facial features.  It may also cause problems if you are training fictional characters with asymmetrical outfits, such as washing out the asymmetries in the outfit.  It is also not suggested if any of your captions included directions like "left" or "right".  Default is 0.0 (no flipping)

## Seed

Seed can be used to make training either more or less deterministic.  The seed value drives both the shuffling of your data set every epoch and also used for your test samples.

To use a random seed, use -1:

    --seed -1

Default behavior is to use a fixed seed of 555. The seed you set is fixed for all samples if you set a value other than -1.  If you set a seed it is also incrememted for shuffling your training data every epoch (i.e. 555, 556, 557, etc).  This makes training more deterministic.  I suggest a fixed seed when you are trying A/B test tweaks to your general training setup, or when you want all your test samples to use the same seed. 

Fixed seed should be used when performing A/B tests or hyperparameter sweeps.  Random seed (-1) may be better if you are stopping and resuming training often so every restart is using random values for all of the various randomness sources used in training such as noising and data shuffling.

## Shuffle tags

For those training booru tagged models, you can use this arg to randomly (but deterministicly unless you use `--seed -1`) all the CSV tags in your captions

    --shuffle_tags ^

This simply chops the captions in to parts based on the commas and shuffles the order. 

In case you want to keep static the first N tags, you can also add this parameter (`--shuffle_tags` must also be set):

    --keep_tags 4 ^

The above example will keep static the 4 first additional tags, and shuffle the rest.

## Zero frequency noise

Based on [Nicholas Guttenberg's blog post](https://www.crosslabs.org//blog/diffusion-with-offset-noise) zero frequency noise offsets the noise added to the image during training/denoising, which can help improve contrast and the ability to render very dark or very bright scenes more accurately, and may help slightly with color saturation.

    --zero_frequency_noise_ratio 0.05

0.0 is off, old behavior.  Default is 0.02 which is very little to err on the safe side (for now?), but values from 0.05 to 0.10 seem to work well.

Test results: https://huggingface.co/panopstor/ff7r-stable-diffusion/blob/main/zero_freq_test_biggs.webp

Very tentatively, I suggest closer to 0.10 for short term training, and lower values of around 0.02 to 0.03 for longer runs (50k+ steps).  Early indications seem to suggest values like 0.10 can cause divergence over time. 

## Keeping images together (custom batching)

If you have a subset of your dataset that expresses the same style or concept, training quality may be improved by putting all of these images through the trainer together in the same batch or batches, instead of the default behaviour (which is to shuffle them randomly throughout the entire dataset).

To control this, put a file called `batch_id.txt` into a folder to give a unique name to the training data in this folder. For example, if you have a bunch of images of horses and you are trying to train them as a single concept, you can assign a unique name such as "my_horses" to these images by putting the word `my_horses` inside `batch_id.txt` in your folder with horse images. 

> Note that because this creates extra aspect ratio buckets, you need to be very careful about correlating the number of images to your training batch size. Aim to have an exact multiple of `batch_size` images at each aspect ratio. For example, if your `batch_size` is 6 and you have images with aspect ratios 4:3, 3:4, and 9:16, you should add or delete images until you have an exact multiple of 6 images (i.e. 6, 12, 28, ...) for each aspect ratio. If you do not do this, the bucketer will duplicate images to fill up each aspect ratio bucket. You'll probably also want to use manual validation to prevent the validator from messing this up, too.

If you are using `.yaml` files for captioning, you can alternatively add a `batch_id: ` entry to either `local.yaml` or the individual images' `.yaml` files. Note that neither `.yaml` nor `batch_id.txt` files act recursively: they do not apply to subfolders.


# Stuff you probably don't need to mess with, but well here it is:


## log_step

Change how often log items are written.  Default is 25 and probably good for most situations.   This does not affect how often samples or ckpts are saved, just how often log scalar items are posted to Tensorboard.

    --log_step 50 ^

Here, the log step is set to a less often "50" number.  Logging has virtually no impact on performance, and there is usually no reason to change this.

## Scale learning rate

Attempts to automatically scale your learning rate up or down base on changes to batch size and gradient accumulation number.

    --scale_lr ^

This multiplies your ```--lr``` setting by ```(batch_size times grad_accum)^0.55```. This can be useful if you're tweaking batch size and grad accum a lot and want to keep your LR to a sane value. 

The value ```0.55``` was derived from the original authors of Stable Diffusion using an LR or 1e-4 for a batch size of 2048 with gradient accumulation 2 (effectively 4096) compared to original Xavier Xiao dreambooth (and forks) commonly using 1e-6 with batch size 1 or 2.  Keep in mind this always *increases* your set ```--lr``` value, so it is suggested to use a lower value for ```--lr``` and let this scale it up, such as ```--lr 2e-6```.  The actual LR used is recorded in your log file and tensorboard and you should pay attention to the logged value as you tweak your batch size and gradient accumulation numbers.  

This is mostly useful for batch size and grad accum tweaking, not for LR tweaking.  Again, watch what actual LR is used to inform your future decisions on LR tweaking. 

## Write batch schedule

If you are interested to see exactly how the images are loaded into batches (i.e. steps), their resolution bucket, and how they are shuffled between epochs, you can use ```--write_schedule``` to output the schedule for every epoch to a file in your log folder.  Keep in mind these files can be large if you are training on a large dataset.  It's not recommended to use this regularly and more of an informational tool for those curious about inner workings of the trainer. 

    --write_schedule ^

The files will be in ```logs/[your project folder]/ep[N]_batch_schedule.txt``` and created every epoch. ex ```ep9_batch_schedule.txt```

## clip_grad_norm

Clips the gradient normals to a maximum value.  Default is None (no clipping).  This is typically used for gradient explosion problems, which are generally not an issue with EveryDream and the grad scaler in AMP mode keeps this from being too much of an issue, but it may be worth experimenting with. 

    --clip_grad_norm 1.0 ^

Default is no gradient normal clipping. There are also other ways to deal with gradient explosion, such as increasing optimizer epsilon.

## Zero Terminal SNR
**Parameter:** `--enable_zero_terminal_snr`  
**Default:** `False`  
To enable zero terminal SNR.

## Dynamic Configuration Loading
**Parameter:** `--load_settings_every_epoch`  
**Default:** `False`  
Most of the parameters in the train.json file CANNOT be modified during training. Activate this to have the `train.json` configuration file reloaded at the start of each epoch. The following parameter can be changed and will be applied after the start of a new epoch:
- `--save_every_n_epochs`
- `--save_ckpts_from_n_epochs`
- `--save_full_precision`
- `--save_optimizer`
- `--zero_frequency_noise_ratio`
- `--min_snr_gamma`
- `--clip_skip`

## Min-SNR-Gamma Parameter
**Parameter:** `--min_snr_gamma`  
**Recommended Values:** 5, 1, 20  
**Default:** `None`  
To enable min-SNR-Gamma. For an in-depth understanding, consult this [research paper](https://arxiv.org/abs/2303.09556).

## EMA Decay Features
The Exponential Moving Average (EMA) model is copied from the base model at the start and is updated every interval of steps by a small contribution from training.
In this mode, the EMA model will be saved alongside the regular checkpoint from training. Normal training checkpoint can be loaded with `--resume_ckpt`, and the EMA model can be loaded with `--ema_decay_resume_model`.
For more information, consult the [research paper](https://arxiv.org/abs/2101.08482) or continue reading the tuning notes below. 
**Parameters:**  
- `--ema_decay_rate`: Determines the EMA decay rate. It defines how much the EMA model is updated from training at each update. Values should be close to 1 but not exceed it. Activating this parameter triggers the EMA decay feature.
- `--ema_strength_target`: Set the EMA strength target value within the (0,1) range. The `ema_decay_rate` is computed based on the relation: decay_rate to the power of (total_steps/decay_interval) equals decay_target. Enabling this parameter will override `ema_decay_rate` and will enable EMA feature. See [ema_strength_target](#ema_strength_target) for more information.
- `--ema_update_interval`: Set the interval in steps between EMA updates. The update occurs at each optimizer step.  If you use grad_accum, actual update interval will be multipled by your grad_accum value.
- `--ema_device`: Choose between `cpu` and `cuda` for EMA. Opting for 'cpu' takes around 4 seconds per update and uses approximately 3.2GB RAM, while 'cuda' is much faster but requires a similar amount of VRAM.
- `--ema_sample_nonema_model`: Activate to display samples from the non-ema trained model, mirroring conventional training. They will not be presented by default with EMA decay enabled.
- `--ema_sample_ema_model`: Turn on to exhibit samples from the EMA model. EMA models will be used for samples generations by default with EMA decay enabled, unless disabled.
- `--ema_resume_model`: Indicate the EMA decay checkpoint to continue from, working like `--resume_ckpt` but will load EMA model. Using `findlast` will only load EMA version and not regular training.

## Notes on tuning EMA.

The purpose of EMA is to reduce the effect of the data from the tail end of training from having an overly powerful effect on the model.  Normally trainig is stopped abruptly and the final images seen by the trainer may have a stronger effect than images seen earlier in training.  *This may have a similar to lowering the learning rate near the end of training, but is not mathematically equivalent.*  An alternative method to EMA would be to use a cosine learning rate schedule. 

Training with EMA turned on has no effect on the non-EMA model if all other settings are identical, though practical considerations (mainly VRAM limits) may cause you to change other settings which can affect the non-EMA model, such as lowering batch size to free enough VRAM for the EMA model if using gpu. 

A standard implementation of EMA uses a decay rate of 0.9999, GPU device, and an interval of 1 (every optimizer step).  This value can have a strong effect, leading to what appears to be an undertrained EMA model compared to the non-EMA model.  A value of 0.999 seems to produce an EMA model nearly identical to the non-EMA model and should be considered a low value.  Somewhere in the 0.999-0.9999 range is suggested when using GPU and interval 1. 

EMA uses an additional ~3.2GB of RAM (for SD1.x models) to store an extra copy of the model weights in memory. For even 24GB consumer GPUs this is substantial, but EMA CPU offloading together with using a higher `ema_update_interval` can make it more practical.  It can be practical on a 24GB GPU if you also enable gradient checkpointing, which is not normally suggested for 24GB GPUs as it is not necessary.  Gradient checkpointing saves a bit more than 3.2GB itself.  The other options is to use CPU offloading by setting `ema_device: "cpu"`.  The EMA copy of the model will be stored in main system memory instead of the GPU, but at a cost of slower sampling and updating.  CPU offloading is a requirement for GPUs with 16GB or less VRAM, and even 24GB GPU users may wish to consider it.  If you are using a 40GB+ GPU you should use GPU. 

When using a higher interval to make cpu offloading practical and reasonably fast, the decay rate should be lowered.  For instance, with an interval of 50, you may wish to lower the decay rate to 0.99 or possibly lower.  This is because the EMA model is updated less frequently and the decay rate is effectively higher than the set value under otherwise "normal" EMA training regime. The higher interval also reduces accuracy of the EMA model compared to the reference implementation which would normally update EMA every optimizer step. 

I would suggest you pick an interval and stick with it, and then tune your decay_rate by generating samples from both EMA and non EMA using the options or after training using your favorite inference app and compare the results. 
It is expected the EMA model will look "behind" on training, but should still be recognizable as the same subject matter.  If it is not, you may wish to try a lower decay rate.  If it is too close to the non-EMA model, you may wish to try a higher decay rate.

Using the GPU for ema incurs only a small speed penalty of around 5-10% with all else being equal, though if you change other parameters such as lowering batch size or enabling gradient checkpointing flag to free VRAM for EMA those options may incur a slightly higher speed penalities. 

Generally, I recommend picking a device and approriate interval given your device choice first and stick with those values, then tweak the `ema_decay_rate` up or down according to how you want the EMA model to look vs. your non-EMA model.  From there, if your EMA model seems to "lag behind" the non-EMA model by "too much" (subjectively judged), you can decrease decay rate. If it identical or nearly identical, use a slightly higher value. 

### ema_strength_target

This arg is a non-standard way of calculating the actual decay rate used. It attempts to calculate a value for decay rate based on your `ema_update_interval` and the total length of training, compensating for both.  Values of 0.01-0.15 should work, with higher values leading to a EMA model that deviates more from the non-EMA model similar to how decay rate works.  It attempts to be more of a "strength" value of EMA, or "how much" (as a factor, i.e. 0.10 = 10% "strength") of the EMA model are kept for the totality of training.  

While the calculation makes sense in how it compensates for inteval and total training length, it is not a standard way of calculating decay rate and there will not be information online about how to use it. I recommend not using this feature and instead picking a device and approriate interval given your device choice first, then tuning your decay rate by hand, find "good" values, then don't mess with them, but you can try this feature out if you want.  

    --ema_strength_target 0.10 ^

If you use `ema_strength_target` the actual calculated `ema_decay_rate` used will be printed in your logs, and you should pay attention to this value and use it to inform your future decisions on EMA tuning.

[Experimental results](https://discord.com/channels/1026983422431862825/1150790432897388556) for general use of EMA on Discord.

## AdaCoor optimizer

This is an optimizer made by stripping out non functional components of CoordinateDoWG and several tweaks to high memory efficiency. It is a learning rate free adaptive optimizer where the only recommended parameter is an epsilon value of 1e-8. This optimizer does not scale well with high batch sizes, so it is recommended to use batch sizes no greater than 8 unless slow and careful training is desired.

## Pyramid_Noise_Discount parameter

This is an implementation of pyramid noise as first introduced here https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2

Pyramid noise can be used to improve dynamic range in short finetunes of < 2000 steps at discounts greater than 0.40. At all discount levels pyramid noise appears to improve the amount of detail generated in images. However, it is not advised to train with pyramid noise for a full training as the noise affects the whole model rapidly and can destabilize the model if trained for too many steps. At 0, pyramid noise is disabled. 

[Experimental results](https://discord.com/channels/1026983422431862825/1176398312870514788) (Discord)

## Attention Type

The `attn_type` arg allows you to select `xformers`, `sdp`, or `slice`.  Xformers uses the [xformers package](https://github.com/facebookresearch/xformers).  SDP uses the scaled dot product mechanism [built into  Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) as of recent Pytorch updates. Slice uses head splitting.  `sdp` is the default and suggested value as it seems to save a small amount of VRAM while also being approximately 5% faster than xformers.  There is likely little reason to use slice or xformers but are kept for the time being for experimentation or consistency with prior experiments.

[Experimental results](https://discord.com/channels/1026983422431862825/1178007113151287306) (Discord link)
