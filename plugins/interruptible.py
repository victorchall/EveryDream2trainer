import math
import os
import shutil
from plugins.plugins import BasePlugin
from train import save_model

EVERY_N_EPOCHS = 1 # how often to save. integers >= 1 save at the end of every nth epoch. floats < 1 subdivide the epoch evenly (eg 0.33 = 3 subdivisions)

class InterruptiblePlugin(BasePlugin):

    def __init__(self):
        print("Interruptible plugin instantiated")
        self.previous_save_path = None
        self.every_n_epochs = EVERY_N_EPOCHS

    def on_epoch_start(self, **kwargs):
        epoch = kwargs['epoch']
        epoch_length = kwargs['epoch_length']
        self.steps_to_save_this_epoch = self._get_save_step_indices(epoch, epoch_length)

    def on_step_end(self, **kwargs):
        local_step = kwargs['local_step']
        if local_step in self.steps_to_save_this_epoch:
            global_step = kwargs['global_step']
            epoch = kwargs['epoch']
            project_name = kwargs['project_name']
            log_folder = kwargs['log_folder']
            ckpt_name = f"rolling-{project_name}-ep{epoch:02}-gs{global_step:05}"
            save_path = os.path.join(log_folder, "ckpts", ckpt_name)
            print(f"{type(self)} saving model to {save_path}")
            save_model(save_path, global_step=global_step, ed_state=kwargs['ed_state'], save_ckpt_dir=None, yaml_name=None, save_ckpt=False, save_full_precision=True, save_optimizer_flag=True)
            self._remove_previous()
            self.previous_save_path = save_path

    def on_training_end(self, **kwargs):
        self._remove_previous()

    def _remove_previous(self):
        if self.previous_save_path is not None:
            shutil.rmtree(self.previous_save_path, ignore_errors=True)
        self.previous_save_path = None

    def _get_save_step_indices(self, epoch, epoch_length_steps: int) -> list[int]:
        if self.every_n_epochs >= 1:
            if ((epoch+1) % self.every_n_epochs) == 0:
                # last step only
                return [epoch_length_steps-1]
            else:
                return []
        else:
            # subdivide the epoch evenly, by rounding self.every_n_epochs to the nearest clean division of steps
            num_divisions = max(1, min(epoch_length_steps, round(1/self.every_n_epochs)))
            # validation happens after training:
            # if an epoch has eg 100 steps and num_divisions is 2, then validation should occur after steps 49 and 99
            validate_every_n_steps = epoch_length_steps / num_divisions
            return [math.ceil((i+1)*validate_every_n_steps) - 1 for i in range(num_divisions)]
