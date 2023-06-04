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

The recommendations are based on "1/10th LR" but "10x the weight decay" compared to AdamW when training diffusion models.  Lion converges quickly, so take care with the learning rate, and even lower learning rates  may be effective.  

There are no known recommendations for the CLIP text encoder.  Using an even larger weight decay, increased epsilon, or even lower LR may be effective for the text encoder.  Further investigation on betas for text encoder is needed as well. 

#### D-Adaption optimizers

[Dadaptation](https://arxiv.org/abs/2301.07733) [version](https://github.com/facebookresearch/dadaptation) of various optimizers.  

These require drastically different hyperparameters.  Early indications seem to point to LR of 0.1 to 1.0 and weight decay of 0.8 may work well.  There is a `decouple` parameter that appears to need to be set to `true` for dadaptation to work and is defaulted. Another `d0` parameter is defaulted to 1e-6 as suggested and, according to the paper authors, does not need to be tuned, but is optional.  See `optimizer_dadapt.json` for an example of a fully configured `dadapt_adam` training. 

These are not memory efficient.  You should use gradient checkpointing even with 24GB GPU.

Available optimizer values for Dadaptation are:

* dadapt_lion, dadapt_adam, dadapt_sgd

These are fairly experimental but tested as working.  Gradient checkpointing may be required even on 24GB GPUs.  Performance is slower than the compiled and optimized AdamW8bit optimizer unless you increae gradient accumulation as it seems the accumulation steps process slowly with the current implementation of D-Adaption

## Optimizer parameters

LR can be set in `optimizer.json` and excluded from the main CLI arg or train.json but if you use the main CLI arg or set it in the main train.json it will override the setting. This was done to make sure existing behavior will not break.  To set LR in the `optimizer.json` make sure to delete `"lr": 1.3e-6` in your main train.json and exclude the CLI arg.

The text encoder LR can run at a different value to the Unet LR. This may help prevent over-fitting, especially if you're training from SD2 checkpoints. 

## Text Encoder freezing

If you're training SD2.1 you will likely experience great benefit from partially freezing the text encoder. You can control text encoder freezing using the `text_encoder_freezing` block in your `optimizer.json`:

```
    "text_encoder_freezing": {
        "freeze_embeddings": true,
        "freeze_front_n_layers": -6,
        "freeze_final_layer_norm": false
    }
```

The SD2.1 text encoder is arranged as follows:

```
embeddings -> CLIP text encoder (23 layers) -> final layer norm
```

(The SD1.5 text encoder is similar but it has only 12 CLIP layers.) Typically you would apply freezing starting from the left and moving to the right, although it might be interesting to experiment with different freezing patterns. You can control this using the following parameters:  

* `freeze_embeddings` freezes the front 2 layers (the text embeddings - recommend). 
* `freeze_front_n_layers` freezes the front N layers of the CLIP text encoder. You can also pass null to leave the CLIP layers unfrozen, or negative values to count from the back. In the example above, `-6` will freeze all but the last 6 layers.
* `freeze_final_layer_norm` freezes the parameters for the text encoder's final `LayerNorm` operation.

Recommended settings for SD2.1 are provided in `optimizerSD21.json`: frozen embeddings, all CLIP layers frozen except for the last 6, final layer norm unfrozen. If you want to experiment, start by trying different values for `freeze_front_n_layers`: `-2` is slower but seems to produce a higher quality model, whereas `-10` is faster but can be more difficult to control. 

## General Beta, weight decay, epsilon, etc tuning

Betas, weight decay, and epsilon are documented in the [AdamW paper](https://arxiv.org/abs/1711.05101) and there is a wealth of information on the web, but consider those experimental to tweak.  
