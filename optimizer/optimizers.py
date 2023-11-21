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
    def __init__(self, args, optimizer_config, text_encoder, unet, epoch_len, log_writer=None):
        del optimizer_config["doc"]
        print(f"\n raw optimizer_config:")
        pprint.pprint(optimizer_config)
        self.epoch_len = epoch_len
        self.unet = unet # needed for weight norm logging, unet.parameters() has to be called again, Diffusers quirk
        self.log_writer = log_writer
        self.te_config, self.base_config = self.get_final_optimizer_configs(args, optimizer_config)
        self.te_freeze_config = optimizer_config.get("text_encoder_freezing", {})
        print(f" Final unet optimizer config:")
        pprint.pprint(self.base_config)
        print(f" Final text encoder optimizer config:")
        pprint.pprint(self.te_config)

        self.grad_accum = args.grad_accum
        self.clip_grad_norm = args.clip_grad_norm
        self.apply_grad_scaler_step_tweaks = optimizer_config.get("apply_grad_scaler_step_tweaks", True)
        self.log_grad_norm = optimizer_config.get("log_grad_norm", True)

        self.text_encoder_params = self._apply_text_encoder_freeze(text_encoder)
        self.unet_params = unet.parameters()

        with torch.no_grad():
            log_action = lambda n, label: logging.info(f"{Fore.LIGHTBLUE_EX} {label} weight normal: {n:.1f}{Style.RESET_ALL}")
            self._log_weight_normal(text_encoder.text_model.encoder.layers.parameters(), "text encoder", log_action)
            self._log_weight_normal(unet.parameters(), "unet", log_action)

        self.optimizers = []
        self.optimizer_te, self.optimizer_unet = self.create_optimizers(args,
                                                                        self.text_encoder_params,
                                                                        self.unet_params)
        self.optimizers.append(self.optimizer_te) if self.optimizer_te is not None else None
        self.optimizers.append(self.optimizer_unet) if self.optimizer_unet is not None else None

        self.lr_schedulers = []
        schedulers = self.create_lr_schedulers(args, optimizer_config)
        self.lr_schedulers.extend(schedulers)

        self.load(args.resume_ckpt)

        self.scaler = GradScaler(
            enabled=args.amp,
            init_scale=2**17.5,
            growth_factor=2,
            backoff_factor=1.0/2,
            growth_interval=25,
        )

        logging.info(f" Grad scaler enabled: {self.scaler.is_enabled()} (amp mode)")

    def _log_gradient_normal(self, parameters: Generator, label: str, log_action=None):
        total_norm = self._get_norm(parameters, lambda p: p.grad.data)
        log_action(total_norm, label)

    def _log_weight_normal(self, parameters: Generator, label: str, log_action=None):
        total_norm = self._get_norm(parameters, lambda p: p.data)
        log_action(total_norm, label)

    def _calculate_normal(param, param_type):
        if param_type(param) is not None:
            return param_type(param).norm(2).item() ** 2
        else:
            return 0.0

    def _get_norm(self, parameters: Generator, param_type):
        total_norm = 0
        for p in parameters:
            param = param_type(p)
            total_norm += self._calculate_norm(param, p)
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _calculate_norm(self, param, p):
        if param is not None:
            return param.norm(2).item() ** 2
        else:
            return 0.0

    def step(self, loss, step, global_step):
        self.scaler.scale(loss).backward()

        if ((global_step + 1) % self.grad_accum == 0) or (step == self.epoch_len - 1):
            if self.clip_grad_norm is not None:
                for optimizer in self.optimizers:
                    self.scaler.unscale_(optimizer)

                if self.log_grad_norm:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(parameters=self.unet.parameters(), max_norm=float('inf'))
                    self.log_writer.add_scalar("optimizer/unet_pre_clip_norm", pre_clip_norm, global_step)

                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder_params, max_norm=float('inf'))
                    self.log_writer.add_scalar("optimizer/te_pre_clip_norm", pre_clip_norm, global_step)

                unet_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.unet.parameters(), max_norm=self.clip_grad_norm)
                self.log_writer.add_scalar("optimizer/unet_grad_norm", unet_grad_norm, global_step)

                te_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder_params, max_norm=self.clip_grad_norm)
                self.log_writer.add_scalar("optimizer/te_grad_norm", te_grad_norm, global_step)

            for optimizer in self.optimizers:
                self.scaler.step(optimizer)

            self.scaler.update()
            if self.log_grad_norm and self.log_writer:
                log_info_unet_fn = lambda n, label: self.log_writer.add_scalar(label, n, global_step)
                log_info_te_fn = lambda n, label: self.log_writer.add_scalar(label, n, global_step)
                with torch.no_grad():
                    self._log_gradient_normal(self.unet_params, "optimizer/unet_grad_norm", log_info_unet_fn)
                    self._log_gradient_normal(self.text_encoder_params, "optimizer/te_grad_norm", log_info_te_fn)

            self._zero_grad(set_to_none=True)

        for scheduler in self.lr_schedulers:
            scheduler.step()

        if self.apply_grad_scaler_step_tweaks:
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

        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler
        base_config["lr_warmup_steps"] = base_config.get("lr_warmup_steps", None) or args.lr_warmup_steps
        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler

        te_config["lr"] = te_config.get("lr", None) or base_config["lr"]
        te_config["optimizer"] = te_config.get("optimizer", None) or base_config["optimizer"]
        te_config["lr_scheduler"] = te_config.get("lr_scheduler", None) or base_config["lr_scheduler"]
        te_config["lr_warmup_steps"] = te_config.get("lr_warmup_steps", None) or  base_config["lr_warmup_steps"]
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
                num_warmup_steps=int(te_config.get("lr_warmup_steps", None) or unet_config.get("lr_warmup_steps",0)),
                num_training_steps=int(te_config.get("lr_decay_steps", None) or unet_config.get("lr_decay_steps",1e9))
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
        import bitsandbytes as bnb
        opt_class = bnb.optim.AdamW8bit
        optimizer = None

        default_lr = 1e-6
        curr_lr = args.lr
        d0 = 1e-6 # dadapt
        decouple = True # seems bad to turn off, dadapt_adam only
        momentum = 0.0 # dadapt_sgd
        no_prox = False # ????, dadapt_adan
        use_bias_correction = True # suggest by prodigy github
        growth_rate=float("inf") # dadapt various, no idea what a sane default is
        safeguard_warmup = True # per recommendation from prodigy documentation

        if local_optimizer_config is not None:
            betas = local_optimizer_config.get("betas", betas)
            epsilon = local_optimizer_config.get("epsilon", epsilon)
            weight_decay = local_optimizer_config.get("weight_decay", weight_decay)
            no_prox =  local_optimizer_config.get("no_prox", False)
            optimizer_name = local_optimizer_config.get("optimizer", "adamw8bit")
            curr_lr = local_optimizer_config.get("lr", curr_lr)
            d0 = local_optimizer_config.get("d0", d0)
            decouple = local_optimizer_config.get("decouple", decouple)
            momentum = local_optimizer_config.get("momentum", momentum)
            growth_rate = local_optimizer_config.get("growth_rate", growth_rate)
            safeguard_warmup = local_optimizer_config.get("safeguard_warmup", safeguard_warmup) 
            if args.lr is not None:
                curr_lr = args.lr
                logging.info(f"Overriding LR from optimizer config with main config/cli LR setting: {curr_lr}")

        if curr_lr is None:
            curr_lr = default_lr
            logging.warning(f"No LR setting found, defaulting to {default_lr}")

        if optimizer_name:
            optimizer_name = optimizer_name.lower()

            if optimizer_name == "lion":
                from lion_pytorch import Lion
                opt_class = Lion
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                )
            elif optimizer_name == "lion8bit":
                from bitsandbytes.optim import Lion8bit
                opt_class = Lion8bit
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                    percentile_clipping=100,
                    min_8bit_size=4096,
                )
            elif optimizer_name == "prodigy":
                from prodigyopt import Prodigy
                opt_class = Prodigy
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    weight_decay=weight_decay,
                    use_bias_correction=use_bias_correction,
                    growth_rate=growth_rate,
                    d0=d0,
                    safeguard_warmup=safeguard_warmup
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
                    raise ValueError(f"Unknown DoWG optimizer {optimizer_name}. Available options are 'coordinate_dowg' and 'scalar_dowg'")
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
                        growth_rate=growth_rate,
                        decouple=decouple,
                    )
                elif optimizer_name == "dadapt_adan":
                    opt_class = dadaptation.DAdaptAdan
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        no_prox=no_prox,
                        weight_decay=weight_decay,
                        eps=epsilon,
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=growth_rate,
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
                        growth_rate=growth_rate,
                    )
            elif optimizer_name == "adacoor":
                from optimizer.adacoor import AdaCoor

                opt_class = AdaCoor
                optimizer = opt_class(
                    itertools.chain(parameters),
                    eps=epsilon
                )

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
        num_layers = len(text_encoder.text_model.encoder.layers)
        unfreeze_embeddings = True
        unfreeze_last_n_layers = None
        unfreeze_final_layer_norm = True
        if "freeze_front_n_layers" in self.te_freeze_config:
            logging.warning(
                ' * Found "freeze_front_n_layers" in JSON, please use "unfreeze_last_n_layers" instead')
            freeze_front_n_layers = self.te_freeze_config["freeze_front_n_layers"]
            if freeze_front_n_layers<0:
                # eg -2 = freeze all but the last 2
                unfreeze_last_n_layers = -freeze_front_n_layers
            else:
                unfreeze_last_n_layers = num_layers - freeze_front_n_layers
        if "unfreeze_last_n_layers" in self.te_freeze_config:
            unfreeze_last_n_layers = self.te_freeze_config["unfreeze_last_n_layers"]

        if unfreeze_last_n_layers is None:
            # nothing specified: default behaviour
            unfreeze_last_n_layers = num_layers
        else:
            # something specified:
            assert(unfreeze_last_n_layers > 0)
            if unfreeze_last_n_layers < num_layers:
                # if we're unfreezing layers then by default we ought to freeze the embeddings
                unfreeze_embeddings = False

        if "freeze_embeddings" in self.te_freeze_config:
            unfreeze_embeddings = not self.te_freeze_config["freeze_embeddings"]
        if "freeze_final_layer_norm" in self.te_freeze_config:
            unfreeze_final_layer_norm = not self.te_freeze_config["freeze_final_layer_norm"]

        parameters = itertools.chain([])

        if unfreeze_embeddings:
            parameters = itertools.chain(parameters, text_encoder.text_model.embeddings.parameters())
        else:
            print(" ❄️ freezing embeddings")

        if unfreeze_last_n_layers >= num_layers:
            parameters = itertools.chain(parameters, text_encoder.text_model.encoder.layers.parameters())
        else:
            # freeze the specified CLIP text encoder layers
            layers = text_encoder.text_model.encoder.layers
            first_layer_to_unfreeze = num_layers - unfreeze_last_n_layers
            print(f" ❄️ freezing text encoder layers 1-{first_layer_to_unfreeze} out of {num_layers} layers total")
            parameters = itertools.chain(parameters, layers[first_layer_to_unfreeze:].parameters())

        if unfreeze_final_layer_norm:
            parameters = itertools.chain(parameters, text_encoder.text_model.final_layer_norm.parameters())
        else:
            print(" ❄️ freezing final layer norm")

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

