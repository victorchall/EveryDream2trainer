# Tweaking settings

Make sure you pay attention to your logs and sample images.  Launch tensorboard in a second command line. See (logging)[doc/LOGGING.md] for more info.

    tensorboard --logdir logs

## Project name

Naming your project will help you track what the heck you're doing when you're floating in checkpoint files later.

You may wish to consider adding "sd1" or "sd2v" or similar to remember what the base was, as you'll also have to tell your inference app what you were using, as its difficult for programs to know what inference YAML to use automatically.  For instance, Automatic1111 webui requires you to copy the v2 inference YAML and rename it to match your checkpoint name so it knows how to load the file, tough it assumes SD 1.x compatible.  Something to keep in mind if you start training on SD2.1.

    --project_name "jets_sd21768v" ^

## Epochs

EveryDream 2.0 has done away with repeats and instead you should set your max_epochs.  Changing epochs has the same effect as changing repeats.  For example, if you had 50 repeats and 5 epochs, you would now set max_epochs to 250.  This is a bit more intuitive as there is no more double meaning for epochs and repeats.

    --max_epochs 250 ^

## Save interval

While EveryDream 1.0 saved a checkpoint every epoch, this is no longer the case as it would produce too many files when repeats are removed.  To balance both people training large and small datasets, you can now set the interval at which checkpoints are saved.  The default is 30 minutes, but you can change it to whatever you want. 

For isntance, if you are working on a very large dataset of thousands of images and lots of different concepts and know it will run for a few hours you may want to save every hour instead, so you would set it to 60.

    --ckpt_every_n_minutes 60 ^

Every save interval, a full ckpt in Diffusers format is saved from which you can continue, and a CKPT format file is also saved for use in your favorite webui.  Keep in mind even save_every_n_epochs 1 is respected, but a pretty bad idea unless you have a lot of disk space...

Additionally, these are saved at the end of training. 

If you wish instead to save every certain number of epochs, you can set the minutes interval 0 and use save_every_n_epochs instead.  This is not recommended for large datasets as it will produce a lot of files.

    --ckpt_every_n_minutes 0 ^
    --save_every_n_epochs 25 ^

## Learning Rate

The learning rate affects how much "training" is done on the model.  It is a very careful balance to select a value that will learn your data, but not overfit it.  If you set the LR too high, the model will "fry" or could "overtrain" and become too rigid, only learning to exactly mimick your training data images and will not be able to generalize to new data or be "stylable".  If you set the LR too low, you may take longer to train, or it may have difficulty learning the concepts at all.  Usually sane values are 1e-6 to 3e-6


## Batch Size

Batch size is also another "hyperparamter" of itself and there are tradeoffs. It may not always be best to use the highest batch size possible.  

While very small batch sizes can impact performance negative, at some point larger sizes have little impact on overall speed.

Larger batch size may also impact what learning rate you use. Often a suggestion is to multiply your LR by the sqrt of batch size.  For example, if you change from batch size 2 to 6, you may consider increasing your LR by sqrt(6/2) or about 1.5x.  This is not a hard rule, but it may help you find a good LR.

    --batch_size 4 ^

## LR Scheduler

A learning rate scheduler can change your learning rate as training progresses.

At this time, ED2.0 supports constant or cosine scheduler. 

The constant scheduler is the default and keeps your LR set to the value you set in the command line.  That's really it for constant!  I recommend sticking with it until you are comfortable with general training.  More info in the [Advanced Tweaking](doc/ATWEAKING.md) document.

## AdamW vs AdamW 8bit

The AdamW optimizer is the default and what was used by EveryDream 1.0.  It's a good optimizer for stable diffusion and appears to be what was used to train SD itself.

AdamW 8bit is quite a bit faster and uses less VRAM.  I currently **recommend** using it for most cases as it seems worth a potential reduction in quality.

    --useadam8bit ^

## Sample prompts

You can set your own sample prompts by adding them, one line at a time, to sample_prompts.txt.  Or you can point to another file with --sample_prompts_file.

    --sample_prompts "project_XYZ_test_prompts.txt" ^

Keep in mind a lot of prompts will take longer to generate.  You may also want to adjust sample_steps to a different value to get samples left often.  This is probably a good idea when training a larger dataset that you know will take longer to train, where more frequent samples will not help you.

    --sample_steps 500 ^

## Log and ckpt save folders

If you want to use a nondefault location for saving logs or ckpt files, these:

Logdir defaults to the "logs" folder in the trainer directory.  If you wan to save all logs (including diffuser copies of ckpts, samples, and tensbooard events) use this:

    --logdir "/workspace/mylogs"

By default the CKPT format copies of ckpts that are peroidically saved are saved in the trainer root folder.  If you want to save them elsewhere, use this:

    --ckpt_dir "r:\webui\models\stable-diffusion"