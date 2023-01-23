# EveryDream 2 low VRAM users guide (<16GB)

A few key arguments will enable training with lower amounts of VRAM.

Amp mode stands for "automatic mixed precision" which allows Torch to choose to execute certain operations that are numerically "safe" to be run in FP16 precision in FP16 precision (addition, subtraction), but use FP32 if unsafe (POW, etc).  This saves some VRAM but also grants a significant performance boost.

    --amp

The next has a significant impact on VRAM.

    --gradient_checkpointing

This enables gradient checkpoint This will reduce the VRAM usage MANY gigabytes. There is a small performance loss, but you can also possible increase your batch size by using it.

The second is batch_size in general.

    --batch_size 1

Keeping the batch_size low reduces VRAM use.  This is a more "fine dial" on VRAM use. Adjusting it up or down by 1 will increase or decrease VRAM use by about 0.5-1GB.  For 12GB gpus you will need to keep batch_size 1 or 2.

The third is gradient accumulation, which does not reduce VRAM, but gives you a "virtual batch size multiplier" when you are not able to increase the batch_size directly.

    --grad_accum 2

This will combine the loss from multiple batches before applying updates.  There is some small VRAM overhead to this but not as much as increasing the batch size.  Increasing it beyond 2 does not continue to increase VRAM, only going from 1 to 2 seems to affect VRAM use, and by a small amount.
