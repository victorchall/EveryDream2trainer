# Advanced Tweaking

## Resolution

You can train resolutions from 512 to 1024 in 64 pixel increments.  General results from the community indicate you can push the base model a bit beyond what it was designed for *with enough training*.  This will work out better when you have a lot of training data (hundreds+) and enable slightly higher resolution at inference time without seeing repeats in your generated images.  This does cost speed of training and higher VRAM use!  Ex. 768 takes a significant amount more VRAM than 512, so you will need to compensate for that by reducing ```batch_size```.

    --resolution 640 ^

For instance, if training from the base 1.5 model, you can try trying at 576, 640, or 704. 

If you are training on a base model that is 768, such as SD 2.1 768-v, you should also probably use 768 as a base number and adjust from there.

## Log and ckpt save folders

If you want to use a nondefault location for saving logs or ckpt files, these:

Logdir defaults to the "logs" folder in the trainer directory.  If you wan to save all logs (including diffuser copies of ckpts, sample images, and tensbooard events) use this:

    --logdir "/workspace/mylogs"

Remember to use the same folder when you launch tensorboard (```tensorboard --logdir "/worksapce/mylogs"```) or it won't find your logs.

By default the CKPT format copies of ckpts that are peroidically saved are saved in the trainer root folder.  If you want to save them elsewhere, use this:

    --ckpt_dir "r:\webui\models\stable-diffusion"

## Conditional dropout

Conditional dropout means the prompt or caption on the training image is dropped, and the caption is "blank".  The theory is this can help with unconditional guidance, per the original paper and authors of Latent Diffusion and Stable Diffusion.

The value is defaulted at 0.04, which means 4% conditional dropout.  You can set it to 0.0 to disable it, or increase it.  Many users of EveryDream 1.0 have had great success tweaking this, especially for larger models.  You may wish to try 0.10.  This may also be useful to really "force" a style.  Setting it very high may lead to bleeding or overfitting.

    --conditional_dropout 0.1 ^

## LR tweaking

By default, the learning rate is constant for the entire training session.  However, if you want it to change by itself during training, you can use cosine.

### Cosine LR scheduler
Cosine LR scheduler will "taper off" your learning rate over time. It will reach a peak value of your ```--lr``` value then taper off following a cosine curve.

Example:

    --lr_scheduler cosine ^

There is also warmup, which will default to 2% of the decay steps.  You can manually set warmup, but it is typically more useful from training a brand new model from scratch, not for continuation training which we're all doing.  But, if you want to tweak manually anyway, use this:

    --lr_warmup_steps 100 ^

Cosine also has a decay period to define how long it takes to get to zero LR as it tapers.  By default, the trainer sets this to slightly longer than it will take to get to your ```--max_epochs``` number of steps.   However, if you want to tweak, you have to set the number of steps yourself and estimate what that will be. 

    --lr_decay_steps 0 ^