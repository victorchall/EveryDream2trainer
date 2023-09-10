import math
import os
import shutil
from plugins.plugins import BasePlugin
from train import save_model

EVERY_N_EPOCHS = 0.3 # how often to save. integers >= 1 save at the end of every nth epoch. floats < 1 subdivide the epoch evenly (eg 0.33 = 3 subdivisions)

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

"""
class InterruptiblePlugin_(BasePlugin):
    def __init__(self, log_folder, args):
        self.log_folder = log_folder
        self.project_name = args.project_name
        self.max_epochs = args.max_epochs

        self.every_n_epochs = 1


    @classmethod
    def make_resume_path(cls, resume_ckpt_folder):
        return os.path.join(resume_ckpt_folder, 'resumable_data.pt')

    def load_resume_state(self, resume_ckpt_path: str, ed_state: EveryDreamTrainingState):
        resume_path = self.make_resume_path(resume_ckpt_path)
        try:
            with open(resume_path, 'rb') as f:
                resumable_data = torch.load(f)
                ed_state.optimizer.load_state_dict(resumable_data['ed_optimizer'])
                ed_state.train_batch.load_state_dict(resumable_data['ed_batch'])
        except Exception as e:
            print(f"InterruptiblePlugin unable to load resume state from {resume_path}: {e}")
            return


    def on_epoch_start(self, ed_state: EveryDreamTrainingState, **kwargs):
        epoch = kwargs['epoch']
        epoch_length = kwargs['epoch_length']
        if epoch == 0:
            resume_ckpt_path = kwargs['resume_ckpt_path']
            self.load_resume_state(resume_ckpt_path, ed_state)
        self.steps_to_save_this_epoch = self._get_save_step_indices(epoch, epoch_length)

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

    def on_step_end(self, epoch: int, local_step: int, global_step: int, ed_state: EveryDreamTrainingState):
        if local_step in self.steps_to_save_this_epoch:
            self.save_and_remove_prior(epoch, global_step, ed_state)

    def _save_and_remove_prior(self, epoch: int, global_step: int, ed_state: EveryDreamTrainingState):
            rolling_save_path = self.make_save_path(epoch, global_step, prepend="rolling-")
            ed_optimizer: EveryDreamOptimizer = ed_state.optimizer
            save_model(rolling_save_path,
                       ed_state=ed_state, save_ckpt_dir=None, yaml_name=None, save_ckpt=False, save_optimizer_flag=True)

                       kwargs['unet'], kwargs['text_encoder'], kwargs['tokenizer'],
                       kwargs['noise_scheduler'], kwargs['vae'], ed_optimizer,
                         save_ckpt_dir=None, yaml_name=None, save_optimizer_flag=True, save_ckpt=False)

            train_batch: EveryDreamBatch = kwargs['train_batch']
            resumable_data = {
                'grad_scaler': ed_optimizer.scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'train_batch': train_batch.state_dict()
            }
            if ed_optimizer.lr_scheduler_te is not None:
                resumable_data['lr_scheduler_te'] = ed_optimizer.lr_scheduler_te.state_dict()
            if ed_optimizer.lr_scheduler_unet is not None:
                resumable_data['lr_scheduler_unet'] = ed_optimizer.lr_scheduler_unet.state_dict()

            torch.save(resumable_data, os.path.join(rolling_save_path, 'resumable_data.pt'))

            self.prev_epoch = epoch
            self.prev_global_step = global_step
        if epoch > 0:
            prev_rolling_save_path = self.make_save_path(epoch, self.prev_global_step, prepend="rolling-")
            shutil.rmtree(prev_rolling_save_path, ignore_errors=True)

        pass

    def make_save_path(self, epoch, global_step, prepend: str="") -> str:
        basename = f"{prepend}{self.project_name}-ep{epoch:02}"
        if global_step is not None:
            basename += f"-gs{global_step:05}"
        return os.path.join(self.log_folder, "ckpts", basename)
"""