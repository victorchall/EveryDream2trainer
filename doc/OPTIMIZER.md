# Advanced optimizer tweaking

You can set advanced optimizer settings using this arg:

    --optimizer_config optimizer.json

or in train.json 

    "optimizer_config": "optimizer.json"

A default `optimizer.json` is supplied which you can modify

This has expanded tweaking.  This doc is incomplete, but there is information on the web on betas and weight decay setting you can search for.

If you do not set optimizer_config, the defaults are `adamw8bit` with standard betas of `(0.9,0.999)`, weight decay `0.01`, and epsilon `1e-8`.  The hyperparameters are originally from XavierXiao's Dreambooth code and based off Compvis Stable Diffusion code. 

## Optimizers

In `optimizer.json` the `optimizer` value is the type of optimizer to use. Below are the supported optimizers.

* adamw

Standard full precision AdamW optimizer exposed by PyTorch.  Not recommended.  Slower and uses more memory than adamw8bit.  Widely documented on the web.

* adamw8bit

Tim Dettmers / bitsandbytes AdamW 8bit optimizer.  This is the default and recommended setting.  Widely documented on the web.

* lion

Lucidrains' [implementation](https://github.com/lucidrains/lion-pytorch) of the [lion optimizer](https://arxiv.org/abs/2302.06675).  Click links to read more.  Unknown what hyperparameters will work well, but paper shows potentially quicker learning.  *Highly experimental, but tested and works.*

## Optimizer parameters

LR can be set in `optimizer.json` and excluded from the main CLI arg or train.json but if you use the main CLI arg or set it in the main train.json it will override the setting. This was done to make sure existing behavior will not break.  To set LR in the `optimizer.json` make sure to delete `"lr": 1.3e-6` in your main train.json and exclude the CLI arg.

Betas, weight decay, and epsilon are documented in the [AdamW paper](https://arxiv.org/abs/1711.05101) and there is a wealth of information on the web, but consider those experimental to tweak.  I cannot provide advice on what might be useful to tweak here.

Note `lion` does not use epsilon.