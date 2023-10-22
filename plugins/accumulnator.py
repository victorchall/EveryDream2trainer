import json
import logging
import os

from plugins.plugins import BasePlugin

class Accumulnator(BasePlugin):

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "accumulnator.json")
        logging.info(f" * Textual Inversion plugin instantiated, loading config from {path}")
        with open(path, 'rt') as f:
            config = json.load(f)
            begin_epoch = config['begin_epoch']
            begin_grad_accum = config['begin_grad_accum']
            end_epoch = config['end_epoch']
            end_grad_accum = config['end_grad_accum']
            accums_per_epoch = {}
            for i in range(begin_epoch):
                accums_per_epoch[i] = begin_grad_accum
            grad_accum_step = (end_grad_accum-begin_grad_accum)/(end_epoch-begin_epoch)
            for i in range(end_grad_accum-begin_grad_accum):
                grad_accum = round(grad_accum_step * i)
                accums_per_epoch[i+begin_epoch] = grad_accum
            self.per_epoch_grad_accum = accums_per_epoch


    def on_epoch_end(self, **kwargs):
        just_finished_epoch = kwargs['epoch']
        epoch = just_finished_epoch + 1
        grad_accum = self.per_epoch_grad_accum.get(epoch)
        if grad_accum is None:
            logging.warning(f" * Acculmunator has no grad_accum setting for epoch {epoch} - leaving as-is")
        else:
            logging.info(f" * Acculmunator setting grad_accum for epoch {epoch} to {grad_accum}")
            arg_update_callback = kwargs['arg_update_callback']
            arg_update_callback('grad_accum', grad_accum)


    def _get_update_step_indices(self, epoch, epoch_length_steps: int) -> list[int]:
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
