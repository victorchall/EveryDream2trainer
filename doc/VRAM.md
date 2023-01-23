# WTF is a CUDA out of memory error?

Training models is very intense on GPU resources, and CUDA out of memory error is quite common and to be expected as you figure out what you can get away with.

## Stuff you want on

Make sure you use the following settings in your json config or command line:

`--amp` on CLI, or in json `"amp": true`

AMP is a significant VRAM savings (and performance increase as well).  It saves several GB and increases performance by 80-100% on Ampere class GPUs.

`--useadam8bit` in CLI or in json `"useadam8bit": true`

Tim Dettmers'  AdamW 8bit optimizer (aka "bitsandbytes") is a significant VRAM savings (and performance increase as well).  Highly recommended, even for high VRAM GPUs.  It saves about 1.5GB and  offers a performance boost.

## I really want to train higher resolution, what do I do?

Gradient checkpointing is pretty useful even on "high" VRAM GPUs like a 24GB 3090 so you can train at 768+ resolution.  While it costs some performance to turn on, it saves a lot of VRAM and can allow you to increase batch size. 

`--gradient_checkpointing` in CLI or in json `"gradient_checkpointing": true`

It is not suggested on 24GB GPUs at 704 or lower resolutoon.  I would keep it off and reduce batch size instead.

Gradient checkpointing is also critical for lower VRAM GPUs like 16 GB T4 (Colab free tier) or 3060 12GB, 2080 Ti 11gb, etc.  You most likely should keep it on for any GPU with less than 24GB and adjust batch size up or down to fit your VRAM.