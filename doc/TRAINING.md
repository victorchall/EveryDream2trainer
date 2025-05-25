# Starting a training session

Here are some example commands to get you started, you can copy paste them into your command line and press enter. 
Make sure the last line does not have ^ but all other lines do.


**First, open a command line, then make sure to activate the environment:**

    activate_venv.bat

You should see your command line show ```(venv)``` at the beginning of the line.  If you don't, something went wrong with setup.

## Running from a json config file

You can edit the example `train.json` file to your liking, then run the following command:

    python train.py --config train.json

Be careful with editing the json file, as any syntax errors will cause the program to crash.  You might want to use a json validator to check your file before running it.  You can use an online validator such as https://jsonlint.com/ or look at it in VS Code.

One particular note is if your path to `data_root` or `resume_ckpt` has backslashes they need to use double \\\ or single /.  There is an example train.json in the repo root.

## Running from the command line with arguments

I recommend you copy one of the examples below and keep it in a text file for future reference.  Your settings are logged in the logs folder, but you'll need to make a command to start training.

Training examples:

Resuming from a checkpoint, 50 epochs, 6 batch size, 3e-6 learning rate, constant scheduler, generate samples evern 200 steps, 10 minute checkpoint interval, and using the default "input" folder for training data:

    python train.py --resume_ckpt "sd_v1-5_vae" ^
    --max_epochs 50 ^
    --data_root "input" ^
    --lr_scheduler constant ^
    --project_name myproj ^
    --batch_size 6 ^
    --sample_steps 200 ^
    --lr 3e-6 ^
    --ckpt_every_n_minutes 10 

Training from SD2 512 base model, 18 epochs, 4 batch size, 1.2e-6 learning rate, constant LR, generate samples evern 100 steps, 30 minute checkpoint interval, using imagesin the x:\mydata folder, training at resolution class of 640:

    python train.py --resume_ckpt "512-base-ema" ^
    --data_root "x:\mydata" ^
    --max_epochs 18 ^
    --lr_scheduler constant ^
    --project_name myproj ^
    --batch_size 4 ^
    --sample_steps 100 ^
    --lr 1.2e-6 ^
    --resolution 640 ^
    --clip_grad_norm 1 ^
    --ckpt_every_n_minutes 30 

Training from the "SD21" model on the "jets" dataset on another drive, for 50 epochs, 6 batch size, 1.5e-6 learning rate, cosine scheduler that will decay in 1500 steps, generate samples evern 100 steps, save a checkpoint every 20 epochs:

    python train.py --resume_ckpt "SD21" ^
    --data_root "R:\everydream-trainer\training_samples\mega\gt\objects\jets" ^
    --max_epochs 25 ^
    --lr_scheduler cosine ^
    --lr_decay_steps 1500 ^
    --lr_warmup_steps 20 ^
    --project_name myproj ^
    --batch_size 6 ^
    --sample_steps 100 ^
    --lr 1.5e-6 ^
    --save_every_n_epochs 20 

Copy paste the above to your command line and press enter.
Make sure the last line does not have ^ but all other lines do.  If you want you can put the command all on one line and not use the ^ carats instead.

## How to resume

Point your resume_ckpt to the path in logs like so:

```--resume_ckpt "R:\everydream2trainer\logs\myproj20221213-161620\ckpts\myproj-ep22-gs01099" ^```

Or use relative pathing:

```--resume_ckpt "logs\myproj20221213-161620\ckpts\myproj-ep22-gs01099" ^```

You should point to the folder in the logs per above if you want to resume rather than running a conversion back on a 2.0GB or 2.5GB pruned file if possible. 

## Rectified Flow and Reflow Training

Rectified Flow is an advanced training technique that aims to straighten the generation trajectory of diffusion models. This can potentially lead to improved sample quality and allow for inference in fewer steps. The "Reflow" mechanism is an iterative process that further refines this flow.

**Important:** Rectified Flow training requires a base model that was originally trained with **velocity prediction** (often referred to as "v-prediction" models, common in Stable Diffusion 2.x series). Using a model not trained with v-prediction will likely lead to poor results. The script will check for this if `--rectified_flow` is enabled.

### Command-Line Arguments:

*   `--rectified_flow`: (boolean, default: `False`)
    Enable Rectified Flow training mode. The model will be trained to predict the direct path from noise to image (velocity).
    *Requires a v-prediction base model.*

*   `--reflow_steps <int>`: (integer, default: `0`)
    Number of Reflow iterations to perform. If `0`, only the initial Rectified Flow training is done (if `--rectified_flow` is set). Each Reflow step involves:
    1.  Generating a new dataset by simulating the current model's flow.
    2.  Retraining the model on this new dataset to further straighten the flow.
    This is an optional enhancement that can iteratively improve the model's generative path.

*   `--reflow_sample_steps <int>`: (integer, default: `0`)
    Number of simulation steps used when generating data during each Reflow iteration. If `0`, the value of `--sample_steps` (used for regular sample generation during training) will be used. If `--sample_steps` is also 0, a default of 50 steps is used.

*   `--reflow_train_epochs <int>`: (integer, default: `1`)
    Number of epochs to train the model on the newly generated dataset during each Reflow iteration.

*   `--reflow_gen_batches <int>`: (integer, default: `1`)
    Number of batches to draw from the original dataset to generate the data for a single Reflow iteration. The total number of generated samples for a reflow iteration will be `reflow_gen_batches * batch_size`.

### Examples:

**Basic Rectified Flow training (no Reflow iterations):**
Ensure `resume_ckpt` points to a v-prediction base model.

```bash
python train.py --config cfgs/train.json ^
--rectified_flow ^
--resume_ckpt path/to/your_v_prediction_model.safetensors
```
(Replace `cfgs/train.json` with your actual config file if different)

**Rectified Flow training with Reflow iterations:**
This example performs 2 Reflow iterations. In each iteration, it uses 10 batches from the original dataset to generate new data and then trains on that new data for 1 epoch.

```bash
python train.py --config cfgs/train.json ^
--rectified_flow ^
--reflow_steps 2 ^
--reflow_gen_batches 10 ^
--reflow_train_epochs 1 ^
--resume_ckpt path/to/your_v_prediction_model.safetensors
```
(Replace `cfgs/train.json` with your actual config file if different)
