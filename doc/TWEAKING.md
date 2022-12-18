# Tweaking settings


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

The constant scheduler is the default and keeps your LR set to the value you set in the command line.  That's really it for constant!

Cosine scheduler will make the LR warm up, then decay along a smooth curve.  The length of the warmup and the length of the "tail" of the curve are defined by additional arguments.

Here's an example of a cosine scheduler 50 step warmup and 1500 step decay:

    --lr 2e-6 ^
    --lr_scheduler cosine ^
    --lr_decay_steps 1500 ^
    --lr_warmup_steps 50 ^

With the above setings, the LR will start at zero, then reach 2e-6 in 50 steps, then decay back to zero in 1500 steps.  If your training continues past 1500 steps, the LR will climb again along a cosine curve, so keep that in mind!

## AdamW vs AdamW 8bit

The AdamW optimizer is the default and what was used by EveryDream 1.0.  It's a good optimizer for stable diffusion and appears to be what was used to train SD itself.

AdamW 8bit is quite a bit faster and uses less VRAM.  I currently do recommend using it for most cases as it seems worth the tradeoff. 

    --useadam8bit ^

