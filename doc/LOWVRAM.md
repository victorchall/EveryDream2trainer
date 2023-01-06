# EveryDream 2 low VRAM users guide (<16GB)

Short version, for 12GB cards, use these arguments:

    --lowvram

This will override various arguments for you to enable training on 12GB cards. 

A few key arguments will enable training with lower amounts of VRAM.

The first is the most impactful.

    --gradient_checkpointing

This enables gradient checkpoint This will reduce the VRAM usage MANY gigabytes.  By itself, and with batch_size 1 VRAM use can be as low as 11.9GB (out of an actual 12.2GB).  This is very tight on a 12GB card such as a 3060 12GB so you will need to take care on what other applications are open. 

The second is batch_size in general.

    --batch_size 1

Keeping the batch_size low reduces VRAM use.  This is a more "fine dial" on VRAM use. Adjusting it up or down by 1 will increase or decrease VRAM use by about 1GB.  For 12GB gpus you will need to keep batch_size 1.

The third is gradient accumulation.

    --grad_accum 2

This will combine the loss from multiple batches before applying updates.  This is like a "virtual batch size multiplier" so if you are limited to just a batch size of 1 or 2 you can increase this to gain some benefits of generalization across multiple images, similar to increasing the batch size.  There is some small VRAM overhead to this, but only when incrementing it from 1 to 2.  If you can run grad_accum 2, you can run 4 or 6.  Your goal here should be to get batch_size times grad_accum to around 8-10.  If you want to try really high values of grad_accum you can, but so far it seems massive batch sizes are not as helpful as you might think.

These are the forced parameters for --lowvram:

    --gradient_checkpointing
    --batch_size 1
    --grad_accum 1
    --resolution 512
