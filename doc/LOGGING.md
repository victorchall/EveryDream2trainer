# Logs

Logs are important to review to track your training and make sure your settings are working as you intend.

Everydream2 uses the Tensorboard library to log performance metrics.  (more options coming!)

You should launch tensorboard while your training is running and watch along.  Open a separate command window, activate venv like you would for training, then run this:

    tensorboard --logdir logs --samples_per_plugin images=100

You can leave Tensorboard running in the background as long as you wish. The `samples_per_plugin` arg will make sure Tensorboard gives finer control on the slider bar for looking through samples, but remember ALL samples are always in your logs, even if you don't see a particular expected sample step in Tensorboard.  

VS Code can also launch Tensorboard by installing the extension, then CTRL-SHIFT-P, start typing "tensorboard" and select "Python:  Launch Tensorboard", "select another folder", and select the "logs" folder under your EveryDream2trainer folder.

## Sample images

Sample images are generated periodically by the trainer to give visual feedback on training progress.  **It's very important to keep an eye on your samples.**  They are available in Tensorboard (and WandB if enabled), or in your logs folder. 

By default, the trainer produces sample images from `sample_prompts.txt` with a fixed seed every so many steps as defined by your `sample_steps` argument.  If you have a ton of them, the slider bar in tensorboard may not select them all (unless you launch tensorboard with the `--samples_per_plugin` argument as shown above), but the actual files are still stored in your logs as well for review.

Samples are produced at CFG scales of 1, 4, and 7. You may find this very useful to see how your model is progressing. 

If your `sample_prompts.txt` is empty, the trainer will generate from prompts from the last batch of your training data, up to 4 sets of samples.

### More control

In place of `sample_prompts.txt` you can provide a `sample_prompts.json` file, which offers more control over sample generation. Here is an example `sample_prompts.json` file:

```json
{
  "batch_size": 3,
  "seed": 555,
  "cfgs": [7, 4],
  "scheduler": "dpm++",
  "num_inference_steps": 15,
  "show_progress_bars": true,
  "generate_samples_every_n_steps": 200,
  "generate_pretrain_samples": true,
  "samples": [
    {
      "prompt": "ted bennet and a man sitting on a sofa with a kitchen in the background",
      "negative_prompt": "distorted, deformed"
    },
    {
      "prompt": "a photograph of ted bennet riding a bicycle",
      "seed": -1,
      "aspect_ratio": 1.77778
    },
    {
      "random_caption": true,
      "size": [640, 384]
    }
  ]
}
```

At the top you can set a `batch_size` (subject to VRAM limits), a default `seed` and `cfgs` to generate with, as well as a `scheduler` and `num_inference_steps` to control the quality of the samples. Available schedulers are `ddim` (the default) and `dpm++`. If you want to see sample progress bars you can set `show_progress_bars` to `true`. To generate a batch of samples before training begins, set `generate_pretrain_samples` to true. 

Finally, you can override the `sample_steps` set in the main configuration .json file (or CLI) by setting `generate_samples_every_n_steps`. This value is read every time samples are updated, so if you initially pass `--sample_steps 200` and then later on you edit your `sample_prompts.json` file to add `"generate_samples_every_n_steps": 100`, after the next set of samples is generated you will start seeing new sets of image samples every 100 steps instead of only every 200 steps.

Individual samples are defined under the `samples` key. Each sample can have a `prompt`, a `negative_prompt`, a `seed` (use `-1` to pick a different random seed each time), and a `size` (must be multiples of 64) or `aspect_ratio` (eg 1.77778 for 16:9). Use `"random_caption": true` to pick a random caption from the training set each time.

## LR

The lr curve is useful to make sure your learning rate curve looks as expected when using something other than constant.  If you hand-tweak the decay steps you may cause issues with the curve, going down and then back up again for instance, in which case you may just wish to remove lr_decay_steps from your command to let the trainer set that for you.

Unet and Text encoder LR are logged separately because the text encoder can be set to ratio of the primary LR.  See [Optimizer](OPTIMIZER.md) for more details.  You can use the logs to confirm the behavior you expect is occurring.

## Loss

Standard loss metrics on Stable Diffusion training jumps around a lot in the scope of the fine tuning the community is doing.  It's not a great metric to use to judge your training unless youa re shooting for a significant shift in the entire model (i.e. training on thousands, tens of thousands, or hundreds of thousands of images in an effort to make a broad shift in what the model generates).

For most users, it's better to look at the samples to subjectively judge if they are improving, or enable [Validation](VALIDATION.md). Validation adds the metric `val/loss` which show meaningful trends. Read the validation documentation for more information and hints on how to intrepet trends in `val/loss`.

## Grad scaler

`hyperparameters/grad scale` is logged for troubleshooting purposes.  If the value trends down to a *negative power* (ex 5e-10), something is wrong with training, such as a wildly inappropriate setting or an issue with your installation.  Otherwise, it bounces around, typically around Ne+3 to Ne+8 and is not much concern. 

## Performance

Images per second will show you when you start a youtube video and your performance tanks.  So, keep an eye on it if you start doing something else on your computer, particularly anything that uses GPU, even playing a video.  

Minutes per epoch is inverse, but you'll see it go up (slower, more minutes per epoch) when there are samples being generated that epoch.  This is normal, but will give you an idea on how your sampling (`--sample_steps`) is affecting your training time.  If you set the sample_steps low, you'll see your minutes per epoch spike more due to the delay involved in generating the samples.  It's still very important to generate samples, but you can weight the cost in speed vs the number of samples.