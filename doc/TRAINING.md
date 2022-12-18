# Starting a training session

Here are some example commands to get you started, you can copy paste them into your command line and press enter. 
Make sure the last line does not have ^ but all other lines do.


**First, open a command line, then make sure to activate the environment:**

    activate_venv.bat

You should see your command line show ```(venv)``` at the beginning of the line.  If you don't, something went wrong with setup.

I recommend you copy one of the examples below and keep it in a text file for future reference.  Your settings are logged in the logs folder, but you'll need to make a command to start training.  

Training examples:

Resuming from a checkpoint, 50 epochs, 6 batch size, 3e-6 learning rate, cosine scheduler, generate samples evern 200 steps, 10 minute checkpoint interval, adam8bit, and using the default "input" folder for training data:

    python train.py --resume_ckpt "sd_v1-5_vae" ^
    --max_epochs 50 ^
    --data_root "input" ^
    --lr_scheduler cosine ^
    --project_name myproj ^
    --batch_size 6 ^
    --sample_steps 200 ^
    --lr 3e-6 ^
    --ckpt_every_n_minutes 10 ^
    --useadam8bit

Training from SD2 512 base model, 18 epochs, 4 batch size, 1.2e-6 learning rate, constant LR, generate samples evern 100 steps, 30 minute checkpoint interval, adam8bit, using imagesin the x:\mydata folder, training at resolution class of 640:

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
    --ckpt_every_n_minutes 30 ^
    --useadam8bit

Training from the "SD21" model on the "jets" dataset on another drive, for 50 epochs, 6 batch size, 1.5e-6 learning rate, cosine scheduler that will decay in 1500 steps, generate samples evern 100 steps, 30 minute checkpoint interval, adam8bit:

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
    --ckpt_every_n_minutes 30 ^
    --useadam8bit 


Copy paste the above to your command line and press enter.
Make sure the last line does not have ^ but all other lines do


Scheduler example, note warmup and decay dont work with constant (default), warmup is set automatically to 5% of decay if not set
--lr_scheduler cosine
--lr_warmup_steps 100
--lr_decay_steps 2500

Warmup and decay only count for some schedulers! Constant is not one of them. 

Currently "constant" and "cosine" are recommended.  I'll add support to others upon request.

## How to resume

Point your resume_ckpt to the path in logs like so:

```--resume_ckpt "R:\everydream2trainer\logs\myproj20221213-161620\ckpts\myproj-ep22-gs01099" ^```
