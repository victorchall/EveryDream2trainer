# Chaining training sessions

EveryDream2 supports "chaining" together multiple training sessions with different settings.

Chaining works by using a magic value for the `resumt_ckpt` argument of `findlast` to find the latest checkpoint. When you use `findlast` the trainer will search your logs for the most recent timestamp diffusers files and use that as the checkpoint to resume from. 

For the first session, you would still need to use a base model, such as `"resume_ckpt": "sd_1-5_vae"` and then copy the config json file and change subsequent sessions to `"resume_ckpt": "findlast"` along with any other modifications you wish to make.

*I highly recommend you modify the `project_name` to make a brief note of the difference.* Perhaps something like `"project_name": "myproject_ch0"` for the first session, `"project_name": "myproject_ch1"` for the second, etc.

Each step in the chain is like its own session, so you get all the same logs but you can walk away from your computer and let it run instead of having to start each session manually.

An example script (bat or bash script) will look like this:

    python train.py --config chain0.json
    python train.py --config chain1.json
    python train.py --config chain2.json

...where the chain json files have modifidations to the previous one, and all **but** the first have `"resume_ckpt": "findlast"`.

You do not need to use the json files, you can also use the command line arguments to chain together sessions, but I generally recommend json files are easier to manage.

You can put the actual python start commands into a .bat or bash script.  Make sure to launch a command window, activate your environment, then run your .bat file from that command window.

NOTE: If one of the sessions crashes, the chain will be broken.  Make sure you are using settings that you know work for your hardware on each session in your chain.

Some ideas for chaining:

* Disable sampling the first session, then enable it on the second session.
    Ex. `"sample_steps": 99999` then `"sample_steps": 300` subsequently.

* Train at lower resolution and higher batch size first to quickly learn concepts, then increase the resolution with a lower batch size to get better quality at higher resolutions.
    Ex. Train at `"resolution": 384` and `"batch_size": 12` first, then `"resolution": 512` and `"batch_size": 7`.

* Train with a lower gradient accumulation value then increase it along with a slightly higher learning rate.
    Ex. start with `"grad_accum": 1` and `"lr": 1e-6` then `"grad_accum": 4` and `"lr": 2e-6`.

* Train with text encoder training enabled first, then disable it using `"disable_textenc_training": true`

* Train a special high aesthetic scoring dataset at 100% conditional dropout `"cond_dropout": 1.0` for a short period (low `max_epochs`), then train your intended dataset at low conditional dropout `"cond_dropout": 0.04`.  Or vice versa. Or alternate them, weaving in brief sessions with the aesthetic set.

* Train with your own learning rate schedule besides cosine or constant by adjusting LR each session by adjusting the `"lr"` each session.

## Examples

Example files are in the root trainer of the folder:

    chain.bat
    chain0.json
    chain1.json
    chain2.json