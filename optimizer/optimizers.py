"""
Copyright [2022-2023] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import itertools
import os
from itertools import chain
from typing import Generator, Any

import torch
from torch.cuda.amp import autocast, GradScaler
from diffusers.optimization import get_scheduler

from colorama import Fore, Style
import pprint

BETAS_DEFAULT = [0.9, 0.999]
EPSILON_DEFAULT = 1e-8
WEIGHT_DECAY_DEFAULT = 0.01
LR_DEFAULT = 1e-6
OPTIMIZER_TE_STATE_FILENAME = "optimizer_te.pt"
OPTIMIZER_UNET_STATE_FILENAME = "optimizer_unet.pt"

class EveryDreamOptimizer():
    """
    Wrapper to manage optimizers
    resume_ckpt_path: path to resume checkpoint, will try to load state (.pt) files if they exist
    optimizer_config: config for the optimizers
    text_encoder: text encoder model parameters
    unet: unet model parameters
    """
    def __init__(self, args, optimizer_config, text_encoder, unet, epoch_len):
        del optimizer_config["doc"]
        print(f"\n raw optimizer_config:")
        pprint.pprint(optimizer_config)
        self.epoch_len = epoch_len
        self.te_config, self.base_config = self.get_final_optimizer_configs(args, optimizer_config)
        self.te_freeze_config = optimizer_config.get("text_encoder_freezing", {})
        print(f" Final unet optimizer config:")
        pprint.pprint(self.base_config)
        print(f" Final text encoder optimizer config:")
        pprint.pprint(self.te_config)

        self.grad_accum = args.grad_accum
        self.clip_grad_norm = args.clip_grad_norm

        self.text_encoder_params = self._apply_text_encoder_freeze(text_encoder)
        self.unet_params = unet.parameters()

        self.optimizers = []
        self.optimizer_te, self.optimizer_unet = self.create_optimizers(args,
                                                                        self.text_encoder_params,
                                                                        self.unet_params)
        self.optimizers.append(self.optimizer_te) if self.optimizer_te is not None else None
        self.optimizers.append(self.optimizer_unet) if self.optimizer_unet is not None else None

        self.lr_schedulers = []
        schedulers = self.create_lr_schedulers(args, optimizer_config)
        self.lr_schedulers.extend(schedulers)
        #print(self.lr_schedulers)

        self.load(args.resume_ckpt)

        self.scaler = GradScaler(
            enabled=args.amp,
            init_scale=2**17.5,
            growth_factor=2,
            backoff_factor=1.0/2,
            growth_interval=25,
        )

        logging.info(f" Grad scaler enabled: {self.scaler.is_enabled()} (amp mode)")

    def step(self, loss, step, global_step):
        self.scaler.scale(loss).backward()

        if ((global_step + 1) % self.grad_accum == 0) or (step == self.epoch_len - 1):
            if self.clip_grad_norm is not None:
                for optimizer in self.optimizers:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=self.unet_params, max_norm=self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder_params, max_norm=self.clip_grad_norm)

            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            
            self.scaler.update()
            self._zero_grad(set_to_none=True)

        for scheduler in self.lr_schedulers:
            scheduler.step()

        self._update_grad_scaler(global_step)

    def _zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_scale(self):
        return self.scaler.get_scale()
    
    def get_unet_lr(self):
        return self.optimizer_unet.param_groups[0]['lr'] if self.optimizer_unet is not None else 0
    
    def get_textenc_lr(self):
        return self.optimizer_te.param_groups[0]['lr'] if self.optimizer_te is not None else 0
    
    def save(self, ckpt_path: str):
        """
        Saves the optimizer states to path
        """
        self._save_optimizer(self.optimizer_te, os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)) if self.optimizer_te is not None else None
        self._save_optimizer(self.optimizer_unet, os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)) if self.optimizer_unet is not None else None

    def load(self, ckpt_path: str):
        """
        Loads the optimizer states from path
        """
        te_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)
        unet_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)
        if os.path.exists(te_optimizer_state_path) and self.optimizer_te is not None:
            self._load_optimizer(self.optimizer_te, te_optimizer_state_path)
        if os.path.exists(unet_optimizer_state_path) and self.optimizer_unet is not None:
            self._load_optimizer(self.optimizer_unet, unet_optimizer_state_path)

    def create_optimizers(self, args, text_encoder_params, unet_params):
        """
        creates optimizers from config and args for unet and text encoder
        returns (optimizer_te, optimizer_unet)
        """

        if args.disable_textenc_training:
            optimizer_te = None
        else:
            optimizer_te = self._create_optimizer("text encoder", args, self.te_config, text_encoder_params)
        if args.disable_unet_training:
            optimizer_unet = None
        else:
            optimizer_unet = self._create_optimizer("unet", args, self.base_config, unet_params)

        return optimizer_te, optimizer_unet

    def get_final_optimizer_configs(self, args, global_optimizer_config):
        """
        defaults and overrides based on priority
        cli LR arg will override LR for both unet and text encoder for legacy reasons
        """
        base_config = global_optimizer_config.get("base")
        te_config = global_optimizer_config.get("text_encoder_overrides")

        if args.lr_decay_steps is None or args.lr_decay_steps < 1:
            # sets cosine so the zero crossing is past the end of training, this results in a terminal LR that is about 25% of the nominal LR
            args.lr_decay_steps = int(self.epoch_len * args.max_epochs * 1.5)

        if args.lr_warmup_steps is None:
            # set warmup to 2% of decay, if decay was autoset to 150% of max epochs then warmup will end up about 3% of max epochs
            args.lr_warmup_steps = int(args.lr_decay_steps / 50)
        
        if args.lr is not None:
            # override for legacy support reasons
            base_config["lr"] = args.lr

        base_config["optimizer"] = base_config.get("optimizer", None) or "adamw8bit"
        base_config["lr_warmup_steps"] = base_config.get("lr_warmup_steps", None) or args.lr_warmup_steps
        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler
        base_config["lr_warmup_steps"] = base_config.get("lr_warmup_steps", None) or args.lr_warmup_steps
        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler

        te_config["lr"] = te_config.get("lr", None) or base_config["lr"]
        te_config["optimizer"] = te_config.get("optimizer", None) or base_config["optimizer"]
        te_config["lr_scheduler"] = te_config.get("lr_scheduler", None) or base_config["lr_scheduler"]
        te_config["lr_warmup_steps"] = te_config.get("lr_warmup_steps", None) or base_config["lr_warmup_steps"]
        te_config["lr_decay_steps"] = te_config.get("lr_decay_steps", None) or base_config["lr_decay_steps"]
        te_config["weight_decay"] = te_config.get("weight_decay", None) or base_config["weight_decay"]
        te_config["betas"] = te_config.get("betas", None) or base_config["betas"]
        te_config["epsilon"] = te_config.get("epsilon", None) or base_config["epsilon"]

        return te_config, base_config

    def create_lr_schedulers(self, args, optimizer_config):        
        unet_config = optimizer_config["base"]
        te_config = optimizer_config["text_encoder_overrides"]

        ret_val = []

        if self.optimizer_te is not None:
            lr_scheduler = get_scheduler(
                te_config.get("lr_scheduler", args.lr_scheduler),
                optimizer=self.optimizer_te,
                num_warmup_steps=te_config.get("lr_warmup_steps", None),
                num_training_steps=unet_config.get("lr_decay_steps", None) or unet_config["lr_decay_steps"]
            )
            ret_val.append(lr_scheduler)    

        if self.optimizer_unet is not None:
            unet_config = optimizer_config["base"]
            lr_scheduler = get_scheduler(
                unet_config["lr_scheduler"],
                optimizer=self.optimizer_unet,
                num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                num_training_steps=int(unet_config["lr_decay_steps"]),
            )
            ret_val.append(lr_scheduler)
        return ret_val

    def _update_grad_scaler(self, global_step):
        if global_step == 500:
            factor = 1.8
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(100)
        if global_step == 1000:
            factor = 1.6
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(200)
        if global_step == 2000:
            factor = 1.3
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(500)
        if global_step == 4000:
            factor = 1.15
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(2000)

    @staticmethod
    def _save_optimizer(optimizer, path: str):
        """
        Saves the optimizer state to specific path/filename
        """
        torch.save(optimizer.state_dict(), path)

    @staticmethod
    def _load_optimizer(optimizer: torch.optim.Optimizer, path: str):
        """
        Loads the optimizer state to an Optimizer object
        optimizer: torch.optim.Optimizer
        path: .pt file
        """
        try:
            optimizer.load_state_dict(torch.load(path))
            logging.info(f" Loaded optimizer state from {path}")
        except Exception as e:
            logging.warning(f"{Fore.LIGHTYELLOW_EX}**Failed to load optimizer state from {path}, optimizer state will not be loaded, \n * Exception: {e}{Style.RESET_ALL}")
            pass

    def _create_optimizer(self, label, args, local_optimizer_config, parameters):
        betas = BETAS_DEFAULT
        epsilon = EPSILON_DEFAULT
        weight_decay = WEIGHT_DECAY_DEFAULT
        opt_class = None
        optimizer = None

        default_lr = 1e-6
        curr_lr = args.lr
        d0 = 1e-6 # dadapt 
        decouple = True # seems bad to turn off, dadapt_adam only
        momentum = 0.0 # dadapt_sgd only

        if local_optimizer_config is not None:
            betas = local_optimizer_config["betas"] or betas
            epsilon = local_optimizer_config["epsilon"] or epsilon
            weight_decay = local_optimizer_config["weight_decay"] or weight_decay
            optimizer_name = local_optimizer_config["optimizer"] or "adamw8bit"
            curr_lr = local_optimizer_config.get("lr", curr_lr)
            d0 = local_optimizer_config.get("d0", d0)
            decouple = local_optimizer_config.get("decouple", decouple)
            momentum = local_optimizer_config.get("momentum", momentum)
            if args.lr is not None:
                curr_lr = args.lr
                logging.info(f"Overriding LR from optimizer config with main config/cli LR setting: {curr_lr}")

        if curr_lr is None:
            curr_lr = default_lr
            logging.warning(f"No LR setting found, defaulting to {default_lr}")

        if optimizer_name:
            if optimizer_name == "lion":
                from lion_pytorch import Lion
                opt_class = Lion
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                )
            elif optimizer_name == "adamw":
                opt_class = torch.optim.AdamW
            if "dowg" in optimizer_name:  
                # coordinate_dowg, scalar_dowg require no additional parameters. Epsilon is overrideable but is unnecessary in all stable diffusion training situations.
                import dowg
                if optimizer_name == "coordinate_dowg":                    
                    opt_class = dowg.CoordinateDoWG
                elif optimizer_name == "scalar_dowg":
                    opt_class = dowg.ScalarDoWG
                else:
                    raise ValueError(f"Unknown DoWG optimizer {optimizer_name}. Available options are coordinate_dowg and scalar_dowg")
            elif optimizer_name in ["dadapt_adam", "dadapt_lion", "dadapt_sgd"]:
                import dadaptation

                if curr_lr < 1e-4:
                    logging.warning(f"{Fore.YELLOW} LR, {curr_lr}, is very low for Dadaptation.  Consider reviewing Dadaptation documentation, but proceeding anyway.{Style.RESET_ALL}")
                if weight_decay < 1e-3:
                    logging.warning(f"{Fore.YELLOW} Weight decay, {weight_decay}, is very low for Dadaptation.  Consider reviewing Dadaptation documentation, but proceeding anyway.{Style.RESET_ALL}")

                if optimizer_name == "dadapt_adam":
                    opt_class = dadaptation.DAdaptAdam
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        weight_decay=weight_decay,
                        eps=epsilon, #unused for lion
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=1e5,
                        decouple=decouple,
                    )
                elif optimizer_name == "dadapt_lion":
                    opt_class = dadaptation.DAdaptLion
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        weight_decay=weight_decay,
                        d0=d0,
                        log_every=args.log_step,
                    )
                elif optimizer_name == "dadapt_sgd":
                    opt_class = dadaptation.DAdaptSGD
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=float("inf"),
                    )

            else:
                import bitsandbytes as bnb
                opt_class = bnb.optim.AdamW8bit

        if not optimizer:
            optimizer = opt_class(
                itertools.chain(parameters),
                lr=curr_lr,
                betas=(betas[0], betas[1]),
                eps=epsilon,
                weight_decay=weight_decay,
                amsgrad=False,
            )

        log_optimizer(label, optimizer, betas, epsilon, weight_decay, curr_lr)
        return optimizer

    def _apply_text_encoder_freeze(self, text_encoder) -> chain[Any]:
        parameters = itertools.chain([])

        if self.te_freeze_config.get('freeze_embeddings', False):
            # freeze embeddings
            print(" ❄️ freezing embeddings")
        else:
            parameters = itertools.chain(parameters, text_encoder.text_model.embeddings.parameters())

        freeze_front_n_layers = self.te_freeze_config.get('freeze_front_n_layers', None)
        if freeze_front_n_layers is None:
            parameters = itertools.chain(parameters, text_encoder.text_model.encoder.layers.parameters())
        else:
            # freeze the specified CLIP text encoder layers
            layers = text_encoder.text_model.encoder.layers
            print(f" ❄️ freezing text encoder layers 0-{len(layers[:freeze_front_n_layers])} of {len(layers)}")
            parameters = itertools.chain(parameters, layers[freeze_front_n_layers:].parameters())

        if self.te_freeze_config.get('freeze_final_layer_norm', False):
            # instead of freezing the final layer norm parameters, we simply do not return them
            print(" ❄️ freezing final layer norm")
        else:
            parameters = itertools.chain(parameters, text_encoder.text_model.final_layer_norm.parameters())

        return parameters


def log_optimizer(label: str, optimizer: torch.optim.Optimizer, betas, epsilon, weight_decay, lr):
    """
    logs the optimizer settings
    """
    all_params = sum([g['params'] for g in optimizer.param_groups], [])
    frozen_parameter_count = len([p for p in all_params if not p.requires_grad])
    total_parameter_count = len(all_params)
    if frozen_parameter_count > 0:
        param_info = f"({total_parameter_count} parameters, {frozen_parameter_count} frozen)"
    else:
        param_info = f"({total_parameter_count} parameters)"

    logging.info(f"{Fore.CYAN} * {label} optimizer: {optimizer.__class__.__name__} {param_info} *{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}    lr: {lr}, betas: {betas}, epsilon: {epsilon}, weight_decay: {weight_decay} *{Style.RESET_ALL}")

