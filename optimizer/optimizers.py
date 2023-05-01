import logging
import itertools
import os

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
    def __init__(self, args, optimizer_config, text_encoder_params, unet_params, epoch_len):
        del optimizer_config["doc"]
        print(f"\noptimizer_config:")
        pprint.pprint(optimizer_config)
        self.grad_accum = args.grad_accum
        self.clip_grad_norm = args.clip_grad_norm
        self.text_encoder_params = text_encoder_params
        self.unet_params = unet_params
        self.epoch_len = epoch_len

        self.optimizer_te, self.optimizer_unet = self.create_optimizers(args, optimizer_config, text_encoder_params, unet_params)
        self.lr_scheduler_te, self.lr_scheduler_unet = self.create_lr_schedulers(args, optimizer_config)

        self.unet_config = optimizer_config.get("unet", {})

        optimizer_te_state_path = os.path.join(args.resume_ckpt, OPTIMIZER_TE_STATE_FILENAME)
        optimizer_unet_state_path = os.path.join(args.resume_ckpt, OPTIMIZER_UNET_STATE_FILENAME)
        if os.path.exists(optimizer_te_state_path):
            logging.info(f"Loading text encoder optimizer state from {optimizer_te_state_path}")
            self.load_optimizer_state(self.optimizer_te, optimizer_te_state_path)
        if os.path.exists(optimizer_unet_state_path):
            logging.info(f"Loading unet optimizer state from {optimizer_unet_state_path}")
            self.load_optimizer_state(self.optimizer_unet, optimizer_unet_state_path)

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

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self.unet_params, max_norm=self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder_params, max_norm=self.clip_grad_norm)
        if ((global_step + 1) % self.grad_accum == 0) or (step == self.epoch_len - 1):
            self.scaler.step(self.optimizer_unet)
            self.scaler.step(self.optimizer_te)
            
            self.scaler.update()
            self._zero_grad(set_to_none=True)

        self.lr_scheduler_unet.step()
        self.lr_scheduler_te.step()
        self._update_grad_scaler(global_step)

    def _zero_grad(self, set_to_none=False):
        self.optimizer_te.zero_grad(set_to_none=set_to_none)
        self.optimizer_unet.zero_grad(set_to_none=set_to_none)
    
    def get_scale(self):
        return self.scaler.get_scale()
    
    def get_unet_lr(self):
        return self.optimizer_unet.param_groups[0]['lr']
    
    def get_textenc_lr(self):
        return self.optimizer_te.param_groups[0]['lr']
    
    def save(self, ckpt_path: str):
        """
        Saves the optimizer states to path
        """
        self._save_optimizer(self.optimizer_te, os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME))
        self._save_optimizer(self.optimizer_unet, os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME))

    def load(self, ckpt_path: str):
        """
        Loads the optimizer states from path
        """
        te_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)
        unet_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)
        if os.path.exists(te_optimizer_state_path):
            self._load_optimizer(self.optimizer_unet, te_optimizer_state_path)
        if os.path.exists(unet_optimizer_state_path):
            self._load_optimizer(self.optimizer_te, unet_optimizer_state_path)

    def create_optimizers(self, args, global_optimizer_config, text_encoder_params, unet_params):
        """
        creates optimizers from config and argsfor unet and text encoder
        returns (optimizer_te, optimizer_unet)
        """
        text_encoder_lr_scale = global_optimizer_config.get("text_encoder_lr_scale")
        unet_config = global_optimizer_config.get("unet")
        te_config = global_optimizer_config.get("text_encoder")
        te_config, unet_config = self.fold(te_config=te_config, 
                                           unet_config=unet_config, 
                                           text_encoder_lr_scale=text_encoder_lr_scale)

        if args.disable_textenc_training:
            optimizer_te = create_null_optimizer()
        else:
            optimizer_te = self.create_optimizer(args, te_config, text_encoder_params)
        if args.disable_unet_training:
            optimizer_unet = create_null_optimizer()
        else:
            optimizer_unet = self.create_optimizer(args, unet_config, unet_params)

        return optimizer_te, optimizer_unet

    @staticmethod
    def fold(te_config, unet_config, text_encoder_lr_scale):
        """
        defaults text encoder config values to unet config values if not specified per property
        """
        if te_config.get("optimizer", None) is None:
            te_config["optimizer"] = unet_config["optimizer"]
        if te_config.get("lr", None) is None:
            te_config["lr"] = unet_config["lr"]
            te_scale = text_encoder_lr_scale
            if te_scale is not None:
                te_config["lr"] = unet_config["lr"] * te_scale
        if te_config.get("weight_decay", None) is None:
            te_config["weight_decay"] = unet_config["weight_decay"]
        if te_config.get("betas", None) is None:
            te_config["betas"] = unet_config["betas"]
        if te_config.get("epsilon", None) is None:
            te_config["epsilon"] = unet_config["epsilon"]
        if te_config.get("lr_scheduler", None) is None:
            te_config["lr_scheduler"] = unet_config["lr_scheduler"]

        return te_config, unet_config

    def create_lr_schedulers(self, args, optimizer_config):
        lr_warmup_steps = int(args.lr_decay_steps / 50) if args.lr_warmup_steps is None else args.lr_warmup_steps
        lr_scheduler_type_unet = optimizer_config["unet"].get("lr_scheduler", None)
        assert lr_scheduler_type_unet is not None, "lr_scheduler must be specified in optimizer config"
        lr_scheduler_type_te = optimizer_config.get("lr_scheduler", lr_scheduler_type_unet)

        if args.lr_decay_steps is None or args.lr_decay_steps < 1:
            args.lr_decay_steps = int(self.epoch_len * args.max_epochs * 1.5)

        lr_warmup_steps = int(args.lr_decay_steps / 50) if args.lr_warmup_steps is None else args.lr_warmup_steps

        self.lr_scheduler_te = get_scheduler(
            lr_scheduler_type_te,
            optimizer=self.optimizer_te,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=args.lr_decay_steps,
        )

        self.lr_scheduler_unet = get_scheduler(
            lr_scheduler_type_unet,
            optimizer=self.optimizer_unet,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=args.lr_decay_steps,
        )

        return self.lr_scheduler_te, self.lr_scheduler_unet

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
            self.scaler.set_growth_interval(1000)
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

    def create_optimizer(self, args, local_optimizer_config, parameters):
        print(f"Creating optimizer from {local_optimizer_config}")
        betas = BETAS_DEFAULT
        epsilon = EPSILON_DEFAULT
        weight_decay = WEIGHT_DECAY_DEFAULT
        opt_class = None
        optimizer = None

        default_lr = 1e-6
        curr_lr = args.lr
        text_encoder_lr_scale = 1.0

        if local_optimizer_config is not None:
            betas = local_optimizer_config["betas"] or betas
            epsilon = local_optimizer_config["epsilon"] or epsilon
            weight_decay = local_optimizer_config["weight_decay"] or weight_decay
            optimizer_name = local_optimizer_config["optimizer"] or None
            curr_lr = local_optimizer_config.get("lr", curr_lr)
            if args.lr is not None:
                curr_lr = args.lr
                logging.info(f"Overriding LR from optimizer config with main config/cli LR setting: {curr_lr}")

            text_encoder_lr_scale = local_optimizer_config.get("text_encoder_lr_scale", text_encoder_lr_scale)
            if text_encoder_lr_scale != 1.0:
                logging.info(f" * Using text encoder LR scale {text_encoder_lr_scale}")

        if curr_lr is None:
            curr_lr = default_lr
            logging.warning(f"No LR setting found, defaulting to {default_lr}")

        curr_text_encoder_lr = curr_lr * text_encoder_lr_scale

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
            elif optimizer_name in ["adamw"]:
                opt_class = torch.optim.AdamW
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

        log_optimizer(optimizer, betas, epsilon, weight_decay, curr_lr, curr_text_encoder_lr)
        return optimizer


def create_null_optimizer():
    return torch.optim.AdamW([torch.zeros(1)], lr=0)

def log_optimizer(optimizer: torch.optim.Optimizer, betas, epsilon, weight_decay, lr, model_name):
    """
    logs the optimizer settings
    """
    logging.info(f"{Fore.CYAN} * Optimizer {model_name}: {optimizer.__class__.__name__} *{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}    lr: {lr}, betas: {betas}, epsilon: {epsilon}, weight_decay: {weight_decay} *{Style.RESET_ALL}")
