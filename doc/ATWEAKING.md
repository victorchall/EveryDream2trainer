# Advanced Tweaking

This document is a bit more geared to experienced users who have trained several models.  It is not required reading for new users.

Start with the [Low VRAM guide](TWEAKING.md) if you are having trouble training on a 12GB card.

## Resolution

You can train resolutions from 512 to 1024 in 64 pixel increments.  General results from the community indicate you can push the base model a bit beyond what it was designed for *with enough training*.  This will work out better when you have a lot of training data (hundreds+) and enable slightly higher resolution at inference time without seeing repeats in your generated images.  This does cost speed of training and higher VRAM use!  Ex. 768 takes a significant amount more VRAM than 512, so you will need to compensate for that by reducing ```batch_size```.

    --resolution 640 ^

For instance, if training from the base 1.5 model, you can try trying at 576, 640, or 704.

If you are training on a base model that is 768, such as "SD 2.1 768-v", you should also probably use 768 as a base number and adjust from there.

Some results from the community seem to indicate training at a higher resolution on SD1.x models may increase how fast the model learns, and it may be a good idea to slightly reduce your learning rate as you increase resolution.  My suspcision is that the higher resolutions increase the gradients as more information is presented to the model per image.  

 You may need to experiment with LR as you increase resolution. I don't have a perfect rule of thumb here, but I might suggest if you train SD1.5 which is a 512 model at resolution 768 you reduce your LR by about half.  ED2 tends to prefer ~2e-6 to ~5e-6 for normal 512 training on SD1.X models around batch 6-8, so if you train SD1.X at 768 consider 1e-6 to 2.5e-6 instead.  

## Log and ckpt save folders

If you want to use a nondefault location for saving logs or ckpt files, these:

Logdir defaults to the "logs" folder in the trainer directory.  If you wan to save all logs (including diffuser copies of ckpts, sample images, and tensbooard events) use this:

    --logdir "/workspace/mylogs"

Remember to use the same folder when you launch tensorboard (```tensorboard --logdir "/worksapce/mylogs"```) or it won't find your logs.

By default the CKPT format copies of ckpts that are peroidically saved are saved in the trainer root folder.  If you want to save them elsewhere, use this:

    --save_ckpt_dir "r:\webui\models\stable-diffusion"

This is useful if you want to dump the CKPT files directly to your webui/inference program model folder so you don't have to manually cut and paste it over.

## Conditional dropout

Conditional dropout means the prompt or caption on the training image is dropped, and the caption is "blank".  The theory is this can help with unconditional guidance, per the original paper and authors of Latent Diffusion and Stable Diffusion.

The value is defaulted at 0.04, which means 4% conditional dropout.  You can set it to 0.0 to disable it, or increase it.  Many users of EveryDream 1.0 have had great success tweaking this, especially for larger models.  You may wish to try 0.10.  This may also be useful to really "force" a style into the model with a high setting such as 0.15.  However, setting it very high may lead to bleeding or overfitting to your training data, especially if your data is not very diverse, which may or may not be desirable for your project.

    --cond_dropout 0.1 ^

## LR tweaking

Learning rate adjustment is a very important part of training.  You can use the default settings, or you can tweak it.  You should consider increasing this further if you increase your batch size further (10+) using [gradient checkpointing](#gradient_checkpointing).

    --lr 1.5e-6 ^

By default, the learning rate is constant for the entire training session.  However, if you want it to change by itself during training, you can use cosine.

## Clip skip

Aka "penultimate layer", this takes the output from the text encoder not from its last output layer, but layers before.  

    --clip_skip 2 ^

A value of "2" is the canonical form of "penultimate layer" useed by various webuis, but 1 to 4 are accepted as well if you wish to experiment.  Default is "0" which takes the "last hidden layer" or standard output of the text encoder as Stable Diffusion 1.X was originally designed.  Training with this setting may necessititate or work better when also using the same setting in your webui/inference program. 

Values of 0 to 3 are valid and working.  The number indicates how many extra layers to go "back" into the CLIP embedding output.  0 is the last layer and the default behavior. 1 is the layer before that, etc.

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

    -- seed -1

Default behavior is to use a fixed seed of 555. The seed you set is fixed for all samples if you set a value other than -1.  If you set a seed it is also incrememted for shuffling your training data every epoch (i.e. 555, 556, 557, etc).  This makes training more deterministic.  I suggest a fixed seed when you are trying A/B test tweaks to your general training setup, or when you want all your test samples to use the same seed. 

## Shuffle tags

For those training booru tagged models, you can use this arg to randomly (but deterministicly unless you use `--seed -1`) all the CSV tags in your captions

    --shuffle_tags ^

This simply chops the captions in to parts based on the commas and shuffles the order. 

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

Clips the gradient normals to a maximum value.  This is an experimental feature, you can read online about gradient clipping.  Default is None (no clipping).  This is typically used for gradient explosion problems, which are not an issue with EveryDream, but might be a fun thing to experiment with?

    --clip_grad_norm 1.0 ^

This may drastically reduce training speed or have other undesirable effects.  My brief toying was mostly unsuccessful.  I would not recommend using this unless you know what you're doing or are bored, but you might discover something cool or interesting.
