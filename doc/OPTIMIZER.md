# Advanced optimizer tweaking

You can set advanced optimizer settings using this arg:

    --optimizer_config optimizer.json

or in train.json 

    "optimizer_config": "optimizer.json"

A default `optimizer.json` is supplied which you can modify.

This extra json file allows expanded tweaking.  

If you do not set `optimizer_config` at all or set it to `null` in train.json, the defaults are `adamw8bit` with standard betas of `(0.9,0.999)`, weight decay `0.01`, and epsilon `1e-8`. 

## Optimizers

In `optimizer.json` the you can set independent optimizer settings for both the text encoder and unet.  If you want shared settings, just fill out the `base` section and leave `text_encoder_overrides` properties null an they will be copied from the `base` section.

If you set the `text_encder_lr_scale` property, the text encoder will be trained with a multiple of the Unet learning rate if it the LR is being copied.  If you explicitly set the text encoder LR, the `text_encder_lr_scale` is ignored.  `text_encder_lr_scale` is likely to be deprecated in the future, but now is left for backwards compatibility. 

For each of the `unet` and `text_encoder` sections, you can set the following properties:

`optimizer` value is the type of optimizer to use. Below are the supported optimizers.

* adamw

Standard full precision AdamW optimizer exposed by PyTorch.  Not recommended.  Slower and uses more memory than adamw8bit.  Widely documented on the web.

* adamw8bit

Tim Dettmers / bitsandbytes AdamW 8bit optimizer.  This is the default and recommended setting.  Widely documented on the web.

* lion

Lucidrains' [implementation](https://github.com/lucidrains/lion-pytorch) of the [lion optimizer](https://arxiv.org/abs/2302.06675).  Click links to read more.  `Epsilon` is not used by lion.

Recommended settings for lion based on the paper are as follows:

    "optimizer": "adamw8bit",
        "lr": 1e-7,
        "lr_scheduler": "constant",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.10

The recommendations are based on "1/10th LR" but "10x the weight decay" compared to AdamW when training diffusion models.  There are no known recommendations for the CLIP text encoder.  Lion converges quickly, so take care with the learning rate, and even lower learning rates  may be effective. 

## Optimizer parameters

LR can be set in `optimizer.json` and excluded from the main CLI arg or train.json but if you use the main CLI arg or set it in the main train.json it will override the setting. This was done to make sure existing behavior will not break.  To set LR in the `optimizer.json` make sure to delete `"lr": 1.3e-6` in your main train.json and exclude the CLI arg.

The text encoder LR can run at a different value to the Unet LR. This may help prevent over-fitting, especially if you're training from SD2 checkpoints. To set the text encoder LR, add a value for `text_encoder_lr_scale` to `optimizer.json` or set the `text_encoder: lr` to its own value (not null). For example, to train the text encoder with an LR that is half that of the Unet, add `"text_encoder_lr_scale": 0.5` to `optimizer.json`. The default value is `0.5`, meaning the text encoder will be trained at half the learning rate of the unet.

## General Beta, weight decay, epsilon, etc tuning

Betas, weight decay, and epsilon are documented in the [AdamW paper](https://arxiv.org/abs/1711.05101) and there is a wealth of information on the web, but consider those experimental to tweak.  
