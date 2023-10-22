import json
import logging
import math
import os
import torch

from plugins.plugins import BasePlugin

class Accumulnator(BasePlugin):

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "accumulnator.json")
        logging.info(f" * Accumulnator plugin instantiated, loading config from {path}")
        with open(path, 'rt') as f:
            config = json.load(f)
            begin_epoch = config['begin_epoch']
            begin_grad_accum = config['begin_grad_accum']
            end_epoch = config['end_epoch']
            end_grad_accum = config['end_grad_accum']

            # spread the grad accums
            curve = config['curve']
            steps = end_epoch - begin_epoch
            if curve == 'linear':
                accums = torch.linspace(start=begin_grad_accum,
                                        end=end_grad_accum,
                                        steps=end_epoch-begin_epoch).tolist()
            elif curve == 'log':
                accums = torch.logspace(start=math.log(begin_grad_accum, 2),
                                        end=math.log(end_grad_accum, 2),
                                        base=2,
                                        steps=steps).tolist()
            else:
                raise NotImplementedError(f"curve not {curve} not recognized")
            #print(f"accums: {accums}")
            accums_per_epoch = {}
            for i in range(begin_epoch):
                accums_per_epoch[i] = begin_grad_accum
            for i in range(steps):
                #print(f"took accum {accums[i]} for epoch {i+begin_epoch}")
                accums_per_epoch[i+begin_epoch] = round(accums[i])

            logging.info(f" * Accumulnator will set grad_accum as follows: {accums_per_epoch}")

            self.per_epoch_grad_accum = accums_per_epoch


    def on_epoch_end(self, **kwargs):
        just_finished_epoch = kwargs['epoch']
        epoch = just_finished_epoch + 1
        grad_accum = self.per_epoch_grad_accum.get(epoch)
        if grad_accum is None:
            logging.warning(f" * Accumulnator has no grad_accum setting for epoch {epoch} - leaving as-is")
        else:
            logging.info(f" * Accumulnator setting grad_accum for epoch {epoch} to {grad_accum}")
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
