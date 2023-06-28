import tkinter as tk
import os
from tkinter import filedialog, messagebox
import subprocess
from tkinter import ttk

def browse_dataset_location():
    dataset_location = filedialog.askdirectory()
    dataset_entry.delete(0, tk.END)
    dataset_entry.insert(tk.END, dataset_location)

def browse_ckpt_location():
    ckpt_location = filedialog.askdirectory()
    ckpt_folder_entry.delete(0, tk.END)
    ckpt_folder_entry.insert(tk.END, ckpt_location)


def launch_training():
    model_type = model_type_combobox.get()
    save_optimizer_state = save_optimizer_state_var.get()
    resume = resume_var.get()
    project_name = project_name_entry.get()
    learning_rate = learning_rate_entry.get()
    match_text_to_unet = match_text_to_unet_var.get()
    text_lr = text_lr_entry.get()
    schedule = schedule_combobox.get()
    text_lr_scheduler = text_lr_scheduler_combobox.get()
    lr_warmup_steps = lr_warmup_steps_entry.get()
    lr_decay_steps = lr_decay_steps_entry.get()
    text_lr_warmup_steps = text_lr_warmup_steps_entry.get()
    text_lr_decay_steps = text_lr_decay_steps_entry.get()
    resolution = resolution_slider.get()
    batch_size = batch_size_entry.get()
    gradient_steps = gradient_steps_entry.get()
    dataset_location = dataset_entry.get()
    max_epochs = max_epochs_slider.get()
    save_every_n_epoch = save_every_n_epoch_entry.get()
    steps_between_samples = steps_between_samples_entry.get()
    training_seed = training_seed_entry.get()
    sample_file = sample_file_combobox.get()
    clip_grad_norm = clip_grad_norm_var.get()
    disable_amp = disable_amp_var.get()
    disable_textenc_training = disable_textenc_training_var.get()
    disable_xformers = disable_xformers_var.get()
    gpuid = gpuid_entry.get()
    gradient_checkpointing = gradient_checkpointing_var.get()
    grad_accum = grad_accum_entry.get()
    logdir = logdir_entry.get()
    log_step = log_step_entry.get()
    lowvram = lowvram_var.get()
    resume_ckpt = Model_to_train.get()
    run_name = run_name_entry.get()
    sample_steps = sample_steps_entry.get()
    save_ckpt_dir = save_ckpt_dir_entry.get()
    save_every_n_epochs = save_every_n_epochs_entry.get()
    save_ckpts_from_n_epochs = save_ckpts_from_n_epochs_entry.get()
    scale_lr = scale_lr_var.get()
    seed = seed_entry.get()
    shuffle_tags = shuffle_tags_var.get()
    validation_config = validation_config_entry.get()
    wandb = wandb_var.get()
    write_schedule = write_schedule_var.get()

    # Construct the command
    command = [
        'python',
        'train.py',
        '--model_type', model_type,
        '--Save_optimizer_state', str(save_optimizer_state),
        '--resume', str(resume),
        '--Project_Name', project_name,
        '--Learning_Rate', learning_rate,
        '--Match_text_to_Unet', str(match_text_to_unet),
        '--Text_lr', text_lr,
        '--Schedule', schedule,
        '--Text_lr_scheduler', text_lr_scheduler,
        '--lr_warmup_steps', lr_warmup_steps,
        '--lr_decay_steps', lr_decay_steps,
        '--Text_lr_warmup_steps', text_lr_warmup_steps,
        '--Text_lr_decay_steps', text_lr_decay_steps,
        '--Resolution', resolution,
        '--Batch_Size', batch_size,
        '--Gradient_steps', gradient_steps,
        '--Dataset_Location', dataset_location,
        '--Max_Epochs', max_epochs,
        '--Save_every_N_epoch', save_every_n_epoch,
        '--Steps_between_samples', steps_between_samples,
        '--Training_Seed', training_seed,
        '--Sample_File', sample_file,
        '--clip_grad_norm', str(clip_grad_norm),
        '--disable_amp', str(disable_amp),
        '--disable_textenc_training', str(disable_textenc_training),
        '--disable_xformers', str(disable_xformers),
        '--gpuid', gpuid,
        '--gradient_checkpointing', str(gradient_checkpointing),
        '--grad_accum', grad_accum,
        '--logdir', logdir,
        '--log_step', log_step,
        '--lowvram', str(lowvram),
        '--resume_ckpt', str(Model_to_train),
        '--run_name', run_name,
        '--sample_steps', sample_steps,
        '--save_ckpt_dir', save_ckpt_dir,
        '--save_every_n_epochs', save_every_n_epochs,
        '--save_ckpts_from_n_epochs', save_ckpts_from_n_epochs,
        '--scale_lr', str(scale_lr),
        '--seed', seed,
        '--shuffle_tags', str(shuffle_tags),
        '--validation_config', validation_config,
        '--wandb', str(wandb),
        '--write_schedule', str(write_schedule),
    ]

    try:
        # Launch the training process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc == 0:
            messagebox.showinfo('Training Complete', 'Training is complete!')
        else:
            messagebox.showerror('Error', f'Training failed with error code {rc}')
    except subprocess.CalledProcessError as e:
        messagebox.showerror('Error', f'Training failed with error: {e}')

# Create the main window
window = tk.Tk()
window.title('EveryDream2 Launch GUI')

# Create a tab control
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Basic Options')
tab_control.add(tab2, text='Advanced Options')
tab_control.add(tab3, text='Logging Options')
tab_control.pack(expand=1, fill='both')

# Create and position the input fields and buttons in tab1 (Basic Options)
model_type_label = tk.Label(tab1, text='Model Type:')
model_type_label.grid(row=0, column=0, sticky=tk.W)

model_type_combobox = ttk.Combobox(tab1, values=['SD2_512_base', 'SD21'])
model_type_combobox.grid(row=0, column=1)
model_type_combobox.current(0)

Model_to_train_label = tk.Label(tab1, text='Model to be trained:')
Model_to_train_label.grid(row=14, column=0, sticky=tk.W)
Model_to_train_entry = tk.Entry(tab3)
Model_to_train_entry.grid(row=14, column=1)

ckpt_folder_label = tk.Label(tab1, text='Checkpoint folder:')
ckpt_folder_label.grid(row=14, column=2, sticky=tk.W)

ckpt_folder_entry = tk.Entry(tab1)
ckpt_folder_entry.grid(row=14, column=1)

browse_ckpt_button = tk.Button(tab1, text='Browse', command=browse_ckpt_location)
browse_ckpt_button.grid(row=14, column=3)


dataset_label = tk.Label(tab1, text='Dataset Location:')
dataset_label.grid(row=1, column=0, sticky=tk.W)

dataset_entry = tk.Entry(tab1)
dataset_entry.grid(row=1, column=1)

browse_button = tk.Button(tab1, text='Browse', command=browse_dataset_location)
browse_button.grid(row=1, column=2)

train_button = tk.Button(tab1, text='Start Training', command=launch_training)
train_button.grid(row=2, column=0, columnspan=3)

resolution_slider_label = tk.Label(tab1, text='Training resolution:')
resolution_slider_label.grid(row=3, column=0, sticky=tk.W)

resolution_slider = tk.Scale(tab1, from_=512, to=1086, resolution=64, orient=tk.HORIZONTAL)
resolution_slider.grid(row=3, column=1)
resolution_slider.set(512)

optimizer_button = tk.Button(tab1, text='Open optimizer.json', command=lambda: os.startfile('optimizer.json'))
optimizer_button.grid(row=4, column=2)

save_optimizer_state_var = tk.BooleanVar()
save_optimizer_state_checkbutton = tk.Checkbutton(tab1, text='Save optimizer state', variable=save_optimizer_state_var)
save_optimizer_state_checkbutton.grid(row=4, column=0, sticky=tk.W)

resume_var = tk.BooleanVar()
resume_checkbutton = tk.Checkbutton(tab1, text='Resume training', variable=resume_var)
resume_checkbutton.grid(row=5, column=0, sticky=tk.W)

project_name_label = tk.Label(tab1, text='Project Name:')
project_name_label.grid(row=6, column=0, sticky=tk.W)

project_name_entry = tk.Entry(tab1)
project_name_entry.grid(row=6, column=1)

batch_size_label = tk.Label(tab1, text='Batch Size:')
batch_size_label.grid(row=7, column=0, sticky=tk.W)

batch_size_entry = tk.Entry(tab1)
batch_size_entry.grid(row=7, column=1)
batch_size_entry.insert(tk.END, '4') # set for testing on my local machine

learning_rate_label = tk.Label(tab1, text='Learning Rate:')
learning_rate_label.grid(row=8, column=0, sticky=tk.W)

learning_rate_entry = tk.Entry(tab1)
learning_rate_entry.grid(row=8, column=1)
learning_rate_entry.insert(tk.END, '1e-6')

match_text_to_unet_var = tk.BooleanVar()
match_text_to_unet_checkbutton = tk.Checkbutton(tab1, text='Match text to Unet', variable=match_text_to_unet_var)
match_text_to_unet_checkbutton.grid(row=9, column=0, sticky=tk.W)

text_lr_label = tk.Label(tab1, text='Text lr:')
text_lr_label.grid(row=10, column=0, sticky=tk.W)

text_lr_entry = tk.Entry(tab1)
text_lr_entry.grid(row=10, column=1)
text_lr_entry.insert(tk.END, '5e-7')

schedule_label = tk.Label(tab1, text='Schedule:')
schedule_label.grid(row=11, column=0, sticky=tk.W)

schedule_combobox = ttk.Combobox(tab1, values=['constant', 'polynomial', 'linear', 'cosine'])
schedule_combobox.grid(row=11, column=1)
schedule_combobox.current(0)

text_lr_scheduler_label = tk.Label(tab1, text='Schedule:')
text_lr_scheduler_label.grid(row=12, column=0, sticky=tk.W)

text_lr_scheduler_label = ttk.Combobox(tab1, values=['constant', 'polynomial', 'linear', 'cosine'])
text_lr_scheduler_label.grid(row=12, column=1)
text_lr_scheduler_label.current(0)

# Create and position the input fields and buttons in tab2 (Advanced Options)
sample_file_label = tk.Label(tab2, text='Sample File:')
sample_file_label.grid(row=0, column=0, sticky=tk.W)

sample_file_combobox = ttk.Combobox(tab2, values=['sample_prompts.json', 'sample_prompts.txt'])
sample_file_combobox.grid(row=0, column=1)
sample_file_combobox.current(1)

open_file_button = tk.Button(tab2, text='Open File', command=lambda: os.startfile(sample_file_combobox.get()))
open_file_button.grid(row=0, column=2)

gradient_steps_label = tk.Label(tab2, text='Gradient Steps:')
gradient_steps_label.grid(row=1, column=0, sticky=tk.W)

gradient_steps_entry = tk.Entry(tab2)
gradient_steps_entry.grid(row=1, column=1)
gradient_steps_entry.insert(tk.END, '4')

clip_grad_norm_var = tk.BooleanVar()
clip_grad_norm_checkbutton = tk.Checkbutton(tab2, text='Clip Grad Norm', variable=clip_grad_norm_var)
clip_grad_norm_checkbutton.grid(row=2, column=0, sticky=tk.W)

disable_amp_var = tk.BooleanVar()
disable_amp_checkbutton = tk.Checkbutton(tab2, text='Disable AMP', variable=disable_amp_var)
disable_amp_checkbutton.grid(row=3, column=0, sticky=tk.W)

disable_textenc_training_var = tk.BooleanVar()
disable_textenc_training_checkbutton = tk.Checkbutton(tab2, text='Disable TextEnc Training', variable=disable_textenc_training_var)
disable_textenc_training_checkbutton.grid(row=4, column=0, sticky=tk.W)

disable_xformers_var = tk.BooleanVar()
disable_xformers_checkbutton = tk.Checkbutton(tab2, text='Disable Xformers', variable=disable_xformers_var)
disable_xformers_checkbutton.grid(row=5, column=0, sticky=tk.W)

gpuid_label = tk.Label(tab2, text='GPU ID:')
gpuid_label.grid(row=6, column=0, sticky=tk.W)

gpuid_entry = tk.Entry(tab2)
gpuid_entry.grid(row=6, column=1)

gradient_checkpointing_var = tk.BooleanVar()
gradient_checkpointing_checkbutton = tk.Checkbutton(tab2, text='Gradient Checkpointing', variable=gradient_checkpointing_var)
gradient_checkpointing_checkbutton.grid(row=7, column=0, sticky=tk.W)

grad_accum_label = tk.Label(tab2, text='Grad Accum:')
grad_accum_label.grid(row=8, column=0, sticky=tk.W)

grad_accum_entry = tk.Entry(tab2)
grad_accum_entry.grid(row=8, column=1)

# Create and position the input fields and buttons in tab3 (Logging Options)
logdir_label = tk.Label(tab3, text='Log Directory:')
logdir_label.grid(row=0, column=0, sticky=tk.W)

logdir_entry = tk.Entry(tab3)
logdir_entry.grid(row=0, column=1)

log_step_label = tk.Label(tab3, text='Log Step:')
log_step_label.grid(row=1, column=0, sticky=tk.W)

log_step_entry = tk.Entry(tab3)
log_step_entry.grid(row=1, column=1)

lowvram_var = tk.BooleanVar()
lowvram_checkbutton = tk.Checkbutton(tab3, text='Low VRAM', variable=lowvram_var)
lowvram_checkbutton.grid(row=2, column=0, sticky=tk.W)

run_name_label = tk.Label(tab3, text='Run Name:')
run_name_label.grid(row=4, column=0, sticky=tk.W)

run_name_entry = tk.Entry(tab3)
run_name_entry.grid(row=4, column=1)

sample_steps_label = tk.Label(tab3, text='Sample Steps:')
sample_steps_label.grid(row=5, column=0, sticky=tk.W)

sample_steps_entry = tk.Entry(tab3)
sample_steps_entry.grid(row=5, column=1)

save_ckpt_dir_label = tk.Label(tab3, text='Save Checkpoint Directory:')
save_ckpt_dir_label.grid(row=6, column=0, sticky=tk.W)

save_ckpt_dir_entry = tk.Entry(tab3)
save_ckpt_dir_entry.grid(row=6, column=1)

save_every_n_epochs_label = tk.Label(tab3, text='Save Every N Epochs:')
save_every_n_epochs_label.grid(row=7, column=0, sticky=tk.W)

save_every_n_epochs_entry = tk.Entry(tab3)
save_every_n_epochs_entry.grid(row=7, column=1)

save_ckpts_from_n_epochs_label = tk.Label(tab3, text='Save Checkpoints from N Epochs:')
save_ckpts_from_n_epochs_label.grid(row=8, column=0, sticky=tk.W)

save_ckpts_from_n_epochs_entry = tk.Entry(tab3)
save_ckpts_from_n_epochs_entry.grid(row=8, column=1)

scale_lr_var = tk.BooleanVar()
scale_lr_checkbutton = tk.Checkbutton(tab3, text='Scale LR', variable=scale_lr_var)
scale_lr_checkbutton.grid(row=9, column=0, sticky=tk.W)

seed_label = tk.Label(tab3, text='Seed:')
seed_label.grid(row=10, column=0, sticky=tk.W)

seed_entry = tk.Entry(tab3)
seed_entry.grid(row=10, column=1)

shuffle_tags_var = tk.BooleanVar()
shuffle_tags_checkbutton = tk.Checkbutton(tab3, text='Shuffle Tags', variable=shuffle_tags_var)
shuffle_tags_checkbutton.grid(row=11, column=0, sticky=tk.W)

validation_config_label = tk.Label(tab3, text='Validation Config:')
validation_config_label.grid(row=12, column=0, sticky=tk.W)

validation_config_entry = tk.Entry(tab3)
validation_config_entry.grid(row=12, column=1)
validation_config_entry.insert(tk.END, 'validation_default.json')

wandb_var = tk.BooleanVar()
wandb_checkbutton = tk.Checkbutton(tab3, text='Wandb', variable=wandb_var)
wandb_checkbutton.grid(row=13, column=0, sticky=tk.W)

write_schedule_var = tk.BooleanVar()
write_schedule_checkbutton = tk.Checkbutton(tab3, text='Write Schedule', variable=write_schedule_var)
write_schedule_checkbutton.grid(row=14, column=0, sticky=tk.W)


# Move the train_button to tab1
train_button.grid(row=17, column=0, columnspan=2, pady=10)

# Start the GUI main loop
window.mainloop()
