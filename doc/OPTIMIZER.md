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
* lion8bit

Tim Dettmers / bitsandbytes AdamW and Lion 8bit optimizer.  adamw8bit is the default and recommended setting as it is well understood, and lion8bit is very vram efficient.  Widely documented on the web.

AdamW vs AdamW8bit: [Experimental results](https://discord.com/channels/1026983422431862825/1120697188427771924) on discord.

* lion

Lucidrains' [implementation](https://github.com/lucidrains/lion-pytorch) of the [lion optimizer](https://arxiv.org/abs/2302.06675).  Click links to read more.  `Epsilon` is not used by lion. You should prefer lion8bit over this optimizer as it is more memory efficient. 

Recommended settings for lion based on the paper are as follows:

    "optimizer": "lion",
        "lr": 1e-7,
        "lr_scheduler": "constant",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.10

The recommendations are based on "1/10th LR" but "10x the weight decay" compared to AdamW when training diffusion models.  Lion converges quickly, so take care with the learning rate, and even lower learning rates  may be effective.  

There are no known recommendations for the CLIP text encoder.  Using an even larger weight decay, increased epsilon, or even lower LR may be effective for the text encoder.  Further investigation on betas for text encoder is needed as well. 

Some investigation on Lion tuning is [here](https://discord.com/channels/1026983422431862825/1098682949978820710) on Discord.

#### D-Adaption optimizers

[Dadaptation](https://arxiv.org/abs/2301.07733) [version](https://github.com/facebookresearch/dadaptation) of various optimizers.  

These require drastically different hyperparameters.  Early indications seem to point to LR of 0.1 to 1.0 and weight decay of 0.8 may work well.  There is a `decouple` parameter that appears to need to be set to `true` for dadaptation to work and is defaulted. Another `d0` parameter is defaulted to 1e-6 as suggested and, according to the paper authors, does not need to be tuned, but is optional.  See `optimizer_dadapt.json` for an example of a fully configured `dadapt_adam` training. 

These are not memory efficient.  You should use gradient checkpointing even with 24GB GPU.

Available optimizer values for Dadaptation are:

* dadapt_lion, dadapt_adam, dadapt_sgd

These are fairly experimental but tested as working.  Gradient checkpointing may be required even on 24GB GPUs.  Performance is slower than the compiled and optimized AdamW8bit optimizer unless you increae gradient accumulation as it seems the accumulation steps process slowly with the current implementation of D-Adaption. 

 #### Prodigy 

Another adaptive optimizer.  It is not very VRAM efficient. [Github](https://github.com/konstmish/prodigy), [Paper](https://arxiv.org/pdf/2306.06101.pdf)

* prodigy

## Optimizer parameters

LR can be set in `optimizer.json` and excluded from the main CLI arg or train.json but if you use the main CLI arg or set it in the main train.json it will override the setting. This was done to make sure existing behavior will not break.  To set LR in the `optimizer.json` make sure to delete `"lr": 1.3e-6` in your main train.json and exclude the CLI arg.

The text encoder LR can run at a different value to the Unet LR. This may help prevent over-fitting, especially if you're training from SD2 checkpoints. 

## Text Encoder freezing

If you're training SD2.1 you will likely experience great benefit from partially freezing the text encoder. You can control text encoder freezing using the `text_encoder_freezing` block in your `optimizer.json`:

```
    "text_encoder_freezing": {
        "unfreeze_final_n_layers": 2,
    }
```

This will freeze the text encoder up to the last 2 layers, leaving the earlier layers and the embeddings intact. 

Recommended settings for SD2.1 are provided in `optimizerSD21.json`. Unfreezing more layers will speed up training at the expense of text encoder stability. You can also try unfreezing the embeddings as well, by setting `"freeze_embeddings": false`. This may improve training, but it also seems to lead to quicker frying. 

There are some [experimental results here](https://discord.com/channels/1026983422431862825/1106511648891609120) (Discord link) on layer freezing.

## General Beta, weight decay, epsilon, etc tuning

Betas, weight decay, and epsilon are documented in the [AdamW paper](https://arxiv.org/abs/1711.05101) and there is a wealth of information on the web, but consider those experimental to tweak.  
