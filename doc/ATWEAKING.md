# Advanced Tweaking

## Resolution

You can train resolutions from 512 to 1024 in 64 pixel increments.  General results from the community indicate you can push the base model a bit beyond what it was designed for *with enough training*.  This will work out better when you have a lot of training data (hundreds+) and enable slightly higher resolution at inference time without seeing repeats in your generated images.  This does cost speed of training and higher VRAM use!  Ex. 768 takes a significant amount more VRAM than 512, so you will need to compensate for that by reducing ```batch_size```.

    --resolution 640 ^

For instance, if training from the base 1.5 model, you can try trying at 576, 640, or 704.

If you are training on a base model that is 768, such as "SD 2.1 768-v", you should also probably use 768 as a base number and adjust from there.

## Log and ckpt save folders

If you want to use a nondefault location for saving logs or ckpt files, these:

Logdir defaults to the "logs" folder in the trainer directory.  If you wan to save all logs (including diffuser copies of ckpts, sample images, and tensbooard events) use this:

    --logdir "/workspace/mylogs"

Remember to use the same folder when you launch tensorboard (```tensorboard --logdir "/worksapce/mylogs"```) or it won't find your logs.

By default the CKPT format copies of ckpts that are peroidically saved are saved in the trainer root folder.  If you want to save them elsewhere, use this:

    --ckpt_dir "r:\webui\models\stable-diffusion"

This is useful if you want to dump the CKPT files directly to your webui/inference program model folder. 

## Clip skip

Aka "penultimate layer", this takes the output from the text encoder not from its last output layer, but layers before.  

    --clip_skip 2 ^

A value of "2" is the canonical form of "penultimate layer" useed by various webuis, but 1 an 3 are accepted as well if you wish to experiment.  Default is "0" which takes the "last hidden layer" or standard output of the text encoder as Stable Diffusion was originally designed.  Training with this setting may necessititate or work better when also using the same setting in your webui/inference program.  

## Conditional dropout

Conditional dropout means the prompt or caption on the training image is dropped, and the caption is "blank".  The theory is this can help with unconditional guidance, per the original paper and authors of Latent Diffusion and Stable Diffusion.

The value is defaulted at 0.04, which means 4% conditional dropout.  You can set it to 0.0 to disable it, or increase it.  Many users of EveryDream 1.0 have had great success tweaking this, especially for larger models.  You may wish to try 0.10.  This may also be useful to really "force" a style.  Setting it very high may lead to bleeding or overfitting.

    --cond_dropout 0.1 ^

## LR tweaking

By default, the learning rate is constant for the entire training session.  However, if you want it to change by itself during training, you can use cosine.

### Cosine LR scheduler
Cosine LR scheduler will "taper off" your learning rate over time. It will reach a peak value of your ```--lr``` value then taper off following a cosine curve.

Example:

    --lr_scheduler cosine ^

There is also warmup, which will default to 2% of the decay steps.  You can manually set warmup, but it is typically more useful from training a brand new model from scratch, not for continuation training which we're all doing.  But, if you want to tweak manually anyway, use this:

    --lr_warmup_steps 100 ^

Cosine also has a decay period to define how long it takes to get to zero LR as it tapers.  By default, the trainer sets this to slightly longer than it will take to get to your ```--max_epochs``` number of steps so LR doesn't go all the way to zero and waste compute time.   However, if you want to tweak, you have to set the number of steps yourself and estimate what that will be.  If you set this, be sure to watch your LR log in tensorboard to make sure it does what you expect.

    --lr_decay_steps 2500 ^

## Gradient accumulation

Gradient accumulation is sort of like a virtual batch size increase, averaging the learning over more than one step (batch) before applying it to the model as an update to weights.

Example:

    --grad_accum 2 ^

The above example with combine the loss from 2 batches before applying updates.  This *may* be a good idea for higher resolution training that requires smaller batch size but mega batch sizes are also not the be-all-end all.

Some experimentation shows if you already have batch size in the 6-8 range than grad accumulation of more than 2 just reduces quality, but you can experiment. 


## Flip_p

If you wish for your training images to be randomly flipped horizontally, use this to flip the images 50% of the time:

    --flip_p 0.5 ^

This is useful for styles or other training that is not symmetrical.  It is not suggested for training specific human faces as it may wash out facial features.  It is also not suggested if any of your captions included directions like "left" or "right".  Default is 0.0 (no flipping)

# Stuff you probably don't need to mess with

## log_step

Change how often log items are written.  Default is 25 and probably good for most situations.   This does not affect how often samples or ckpts are saved, just log scalar items. 

    --log_step 50 ^

## scale_lr

Attempts to automatically scale your learning rate up or down base on changes to batch size and gradient accumulation.

    --scale_lr ^

This multiplies your ```--lr``` setting by ```sqrt of (batch_size times grad_accum)```. This can be useful if you're tweaking batch size and grad accum and want to keep your LR to a sane value. 

## clip_grad_norm

Clips the gradient normals to a maximum value.  This is an experimental feature, you can read online about gradient clipping.  Default is None (no clipping).  This is typically used for gradient explosion problems, but might be a fun thing to experiment with.

    --clip_grad_norm 1.0 ^