# __Tweaking settings__

This document should be read by all users who are trying to get the best results out of EveryDream 2.0.  These are the key settings you'll need to understand to get started.

## __Logging__

Make sure you pay attention to your logs and sample images.  Launch tensorboard in a second command line. See [logging](LOGGING.md) for more info.

    tensorboard --logdir logs

## __Project name__

Naming your project will help you track what the heck you're doing when you're floating in checkpoint files later.

You may wish to consider adding "sd1" or "sd2v" or similar to remember what the base was, as you'll also have to tell your inference app what you were using, as its difficult for programs to know what inference YAML to use automatically.  For instance, Automatic1111 webui requires you to copy the v2 inference YAML and rename it to match your checkpoint name so it knows how to load the file, tough it assumes SD 1.x compatible.  Something to keep in mind if you start training on SD2.1.

    --project_name "jets_sd21768v" ^


## __Stuff you probably want on__

### amp
    --amp

Enables automatic mixed precision.  Greatly improved training speed and will reduce VRAM use.  [Torch](https://pytorch.org/docs/stable/amp.html) will automatically use FP16 precision for specific model components where FP16 is sufficient precision, and FP32 otherwise.  This also enables xformers to work with the SD1.x attention head schema, which is a large speed boost for SD1.x training.  I highly suggest you always use this, but it is left as an option if you wish to disable.

When amp is used with [gradient checkpointing](#gradient_checkpointing) you can run the trainer on 12GB GPUs and potentially 11GB.

### useadam8bit 

    --useadam8bit

Uses [Tim Dettmer's reduced precision AdamW 8 Bit optimizer](https://github.com/TimDettmers/bitsandbytes).  This seems to have no noticeable impact on quality but is considerable faster and more VRAM efficient. See more below in AdamW vs AdamW 8bit.

## __Epochs__

EveryDream 2.0 has done away with repeats and instead you should set your max_epochs.  Changing epochs has the same effect as changing repeats in DreamBooth or EveryDream1.  For example, if you had 50 repeats and 5 epochs, you would now set max_epochs to 250 (50x5=250).  This is a bit more intuitive as there is no more double meaning for epochs and repeats.

    --max_epochs 250 ^

This is like your "amount" of training.  

With more training data for your subjects and concepts, you can slowly scale this value down.  More example images mean an epoch is longer, and more training is done simply by the fact there is more training data.

With less training data, this value should be higher, because more repetition on the images is needed to learn.

## __Resolution__

The resolution for training.  All buckets for multiaspect will be based on the total pixel count of your resolution squared. 

    --resolution 768

Current supported resolutions can be printed by running the trainer without any arugments.

    python train.py

## __Save interval for checkpoints__

While EveryDream 1.0 saved a checkpoint every epoch, this is no longer the case as it would produce too many files as "repeats" are removed in favor of just using epochs instead.  To balance the fact EveryDream users are sometimes training small datasets and sometimes huge datasets, you can now set the interval at which checkpoints are saved.  The default is 30 minutes, but you can change it to whatever you want. 

For isntance, if you are working on a very large dataset of thousands of images and lots of different concepts and know it will run for a few hours you may want to save every hour instead, so you would set it to 60.

    --ckpt_every_n_minutes 60 ^

Every save interval, a full ckpt in Diffusers format is saved from which you can continue, and a CKPT format file is also saved for use in your favorite webui.  Keep in mind even save_every_n_epochs 1 is respected, but a pretty bad idea unless you have a lot of disk space...

Additionally, these are saved at the end of training. 

If you wish instead to save every certain number of epochs, save_every_n_epochs instead.  

    --save_every_n_epochs 25 ^

If you are training a huge dataset (20k+) then saving every 1 epoch may not be very often, so consider using ckpt_every_n_minutes as mentioned above instead.

*A "last" checkpoint is always saved at the end of training.*

Diffusers copies of checkpoints are saved in your /logs/[project_name]/ckpts folder, and can be used to continue training if you want to pick up where you left off.  CKPT files are saved in the root training folder by default.  These folders can be changed. See [Advanced Tweaking](ATWEAKING.md) for more info.

## __Resuming training from previous runs__

If you want to resume training from a previous run, you can do so by pointing to the diffusers copy in the logs folder from which you want to resume.  This is the same --resume_ckpt argument you would use to start training, just pointing to a different location.

    --resume_ckpt "logs\city_gradckptng2_20221231-234604\ckpts\last-city_gradckptng2-ep59-gs00600" ^

## __Learning Rate__

The learning rate affects how much "training" is done on the model per training step.  It is a very careful balance to select a value that will learn your data.  See [Advanced Tweaking](ATWEAKING.md) for more info.  Once you have started, the learning rate is a good first knob to turn as you move into more advanced tweaking.

## __Batch Size__

Batch size is also another "hyperparamter" of itself and there are tradeoffs. It may not always be best to use the highest batch size possible.  Once of the primary reasons to change it is if you get "CUDA out of memory" errors where lowering the value may help.

    --batch_size 4 ^

While very small batch sizes can impact performance negatively, at some point larger sizes have little impact on overall speed as well, so shooting for the moon is not always advisable.  Changing batch size may also impact what learning rate you use, with typically larger batch_size requiring a slightly higher learning rate.  More info is provided in the [Advanced Tweaking](ATWEAKING.md) document.

## __LR Scheduler__

A learning rate scheduler can change your learning rate as training progresses.

At this time, ED2.0 supports constant or cosine scheduler. 

The constant scheduler is the default and keeps your LR set to the value you set in the command line.  That's really it for constant!  I recommend sticking with it until you are comfortable with general training.  More info in the [Advanced Tweaking](ATWEAKING.md) document.

## __Sampling__

You can set your own sample prompts by adding them, one line at a time, to sample_prompts.txt.  Or you can point to another file with --sample_prompts.

    --sample_prompts "project_XYZ_test_prompts.txt" ^

Keep in mind a longer list of prompts will take longer to generate.  You may also want to adjust sample_steps to a different value to get samples left often.  This is probably a good idea when training a larger dataset that you know will take longer to train, where more frequent samples will not help you.

Sample steps declares how often samples are generated and put into the logs and Tensorboard.

    --sample_steps 300 ^

Keep in mind if you drastically change your batch_size, the frequency (in time between samples) of samples will change.  Going from batch size 2 to batch size 10 may reduce how fast steps process, so you may want to reduce sample_steps to compensate.

## __Gradient checkpointing__

This is mostly useful to reduce VRAM for smaller GPUs, and together with AdamW 8 bit and AMP mode can enable <12GB GPU training.

Gradient checkpointing can also offer a higher batch size and/or higher resolution within whatever VRAM you have, so it may be useful even on a 24GB+ GPU if you specifically want to run a very large batch size.  The other option is using gradient accumulation instead.

    --gradient_checkpointing ^

While gradient checkpointing reduces performance, the ability to run a higher batch size brings performance back fairly close to without it. 

You may NOT want to use a batch size as large as 13-14+ on your 24GB+ GPU even if possible, or you may find you need to tweak learning rate all over again to find the right balance.  Generally I would not turn it on for a 24GB GPU training at <640 resolution.

This probably IS a good idea for training at higher resolutions and allows >768 training on 24GB GPUs.  Balancing this toggle, resolution, and batch_size will take a few quick experiments to see what you can run safely.