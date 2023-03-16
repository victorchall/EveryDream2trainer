# WTF is a CUDA out of memory error?

Training models is very intense on GPU resources, and `CUDA out of memory error` is quite common and to be expected as you figure out what you can get away with inside the constraints of your GPU VRAM limit.

VRAM use depends on the model being trained (SD1.5 vs SD2.1 base), batch size, resolution, and a number of other settings.

## Stuff you want on for 12GB cards

AMP and AdamW8bit are now defaulted to on.  These are VRAM efficient, produce high quality results, and should be on for all training.

Gradient checkpointing can still be turned on and off, and is not on by default.  Turning it on will greatly reduce VRAM use at the expense of some performance.  It is suggested to turn it on for any GPU with less than 16GB VRAM and *is definitely required for 12GB cards*.

If you are using a customized `optimizer.json`, make sure `adamw8bit` is set as the optimizer.  `AdamW` is significantly more VRAM intensive. `lion` is another option that is VRAM efficient, but is still fairly experimental in terms of understanding the best LR, betas, and weight decay settings.  See [Optimizer docs](OPTIMIZER.md) for more information on advanced optimizer config if you want to try `lion` optimizer. *`adamw8bit` is the recommended and also the default.*

SD2.1 with the larger text encoder model may not train on 12GB cards.  SD1.5 should work fine.

Batch size of 1 or 2 may be all you can use on 12GB.

Resolution of 512 may be all you can use on 12GB.  You could try 576 or 640 at batch size 1.

Due to other things running on any given users' systems, precise advice cannot be given on what will run, though 12GB certainly can and does work.

Close all other programs and processes that are using GPU resources.  Apps like Chrome and Discord can use many hundreds of megabytes of VRAM, and can add up quickly. You can also try disabling "hardware acceleration" in some apps which will shift the resources to CPU and system RAM, and save VRAM.

## I really want to train higher resolution, what do I do?

Gradient checkpointing is pretty useful even on "high" VRAM GPUs like a 24GB 3090 so you can train at 768+ resolution.  While it costs some performance to turn on, it saves a lot of VRAM and can allow you to increase batch size. 

`--gradient_checkpointing` in CLI or in json `"gradient_checkpointing": true`

It is not suggested on 24GB GPUs at 704 or lower resolutoon.  I would keep it off and reduce batch size instead to fit your training into VRAM.

