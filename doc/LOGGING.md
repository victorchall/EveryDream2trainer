# Logs

Logs are important to review to track your training and make sure your settings are working as you intend.

Everydream2 uses the Tensorboard library to log performance metrics.  (more options coming!)

You should launch tensorboard while your training is running and watch along.

    tensorboard --logdir logs --samples_per_plugin images=100

## Sample images

By default, the trainer produces sample images from `sample_prompts.txt` with a fixed seed every so many steps as defined by your `sample_steps` argument. These are saved in the logs directory and can be viewed in tensorboard as well if you prefer. If you have a ton of them, the slider bar in tensorboard may not select them all (unless you launch tensorboard with the `--samples_per_plugin` argument as shown above), but the actual files are still stored in your logs as well for review.

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

At the top you can set a `batch_size` (subject to VRAM limits), a default `seed` and `cfgs` to generate with, as well as a `scheduler` and `num_inference_steps` to control the quality of the samples. Available schedulers are `ddim` (the default) and `dpm++`. Finally, you can set `show_progress_bars` to `true` if you want to see progress bars during the sample generation process. 

Individual samples are defined under the `samples` key. Each sample can have a `prompt`, a `negative_prompt`, a `seed` (use `-1` to pick a different random seed each time), and a `size` (must be multiples of 64) or `aspect_ratio` (eg 1.77778 for 16:9). Use `"random_caption": true` to pick a random caption from the training set each time.

## LR

The lr curve is useful to make sure your learning rate curve looks as expected when using something other than constant.  If you hand-tweak the decay steps you may cause issues with the curve, going down and then back up again for instance, in which case you may just wish to remove lr_decay_steps from your command to let the trainer set that for you.

## Loss

To be perfectly honest, loss on stable diffusion training just jumps around a lot.  It's not a great metric to use to judge your training.  It's better to look at the samples and see if they are improving.

## Performance

Images per second will show you when you start a youtube video and your performance tanks.  So, keep an eye on it if you start doing something else on your computer, particularly anything that uses GPU, even playing a video.  Note that the initial performance has a ramp up time, once it gets going it should maintain as long as you don't do anything else that uses GPU.  I have occasionally had issues with my GPU getting "locked" into "slow mode" after trying to play a video, so watch out for that.

Minutes per epoch is inverse, but you'll see it go up (slower, more minutes per epoch) when there are samples being generated that epoch.  This is normal, but will give you an idea on how your sampling (``--sample_steps``) is affecting your training time.  If you set the sample_steps low, you'll see your minutes per epoch spike more due to the delay involved in generating.  It's still very important to generate samples, but you can weight the cost in speed vs the number of samples.