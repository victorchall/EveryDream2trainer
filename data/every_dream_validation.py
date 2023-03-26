import json
import logging
import math
import random
from typing import Callable, Any, Optional, Generator
from argparse import Namespace

import torch
import numpy as np
from colorama import Fore, Style
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from data.data_loader import DataLoaderMultiAspect
from data import resolver
from data import aspects
from data.image_train_item import ImageTrainItem
from utils.isolate_rng import isolate_rng


def get_random_split(items: list[ImageTrainItem], split_proportion: float, batch_size: int) \
        -> tuple[list[ImageTrainItem], list[ImageTrainItem]]:
    split_item_count = max(1, math.ceil(split_proportion * len(items)))
    # sort first, then shuffle, to ensure determinate outcome for the current random state
    items_copy = list(sorted(items, key=lambda i: i.pathname))
    random.shuffle(items_copy)
    split_items = list(items_copy[:split_item_count])
    remaining_items = list(items_copy[split_item_count:])
    return split_items, remaining_items

def disable_multiplier_and_flip(items: list[ImageTrainItem]) -> Generator[ImageTrainItem, None, None]:
    for i in items:
        yield ImageTrainItem(image=i.image, caption=i.caption, aspects=i.aspects, pathname=i.pathname, flip_p=0, multiplier=1)

class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 default_batch_size: int,
                 resolution: int,
                 log_writer: SummaryWriter,
    ):
        self.val_dataloader = None
        self.train_overlapping_dataloader = None
        self.resolution = resolution
        self.log_writer = log_writer

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,

            'validate_training': True,
            'val_split_mode': 'automatic',
            'val_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15,

            'use_relative_loss': False,
        }
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                self.config.update(json.load(f))

        self.train_overlapping_dataloader_loss_offset = None
        self.val_loss_offset = None

        self.loss_val_history = []
        self.val_loss_window_size = 5 # todo: arg for this?

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def every_n_epochs(self):
        return self.config['every_n_epochs']

    @property
    def seed(self):
        return self.config['seed']
    
    @property
    def use_relative_loss(self):
        return self.config['use_relative_loss']

    def prepare_validation_splits(self, train_items: list[ImageTrainItem], tokenizer: Any) -> list[ImageTrainItem]:
        """
        Build the validation splits as requested by the config passed at init.
        This may steal some items from `train_items`.
        If this happens, the returned `list` contains the remaining items after the required items have been stolen.
        Otherwise, the returned `list` is identical to the passed-in `train_items`.
        """
        with isolate_rng():
            random.seed(self.seed)
            self.val_dataloader, remaining_train_items = self._build_val_dataloader_if_required(train_items, tokenizer)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            self.train_overlapping_dataloader = self._build_train_stabilizer_dataloader_if_required(
                remaining_train_items, tokenizer)
            return remaining_train_items

    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                mean_loss = self._calculate_validation_loss('stabilize-train',
                                                            self.train_overlapping_dataloader,
                                                            get_model_prediction_and_target_callable)
                if self.train_overlapping_dataloader_loss_offset is None:
                    self.train_overlapping_dataloader_loss_offset = -mean_loss
                self.log_writer.add_scalar(tag=f"loss/stabilize-train",
                                           scalar_value=self.train_overlapping_dataloader_loss_offset + mean_loss,
                                           global_step=global_step)
            if self.val_dataloader is not None:
                mean_loss = self._calculate_validation_loss('val',
                                                            self.val_dataloader,
                                                            get_model_prediction_and_target_callable)
                if self.val_loss_offset is None:
                    self.val_loss_offset = -mean_loss
                self.log_writer.add_scalar(tag=f"loss/val",
                                           scalar_value=mean_loss if not self.use_relative_loss else self.val_loss_offset + mean_loss,
                                           global_step=global_step)
                

                self.track_loss_trend(mean_loss)

    def track_loss_trend(self, mean_loss):
        self.loss_val_history.append(mean_loss)

        if len(self.loss_val_history) > ((self.val_loss_window_size * 2) + 1):
            dy = np.diff(self.loss_val_history[-self.val_loss_window_size:])
            if np.average(dy) > 0:
                logging.warning(f"Validation loss shows diverging.  Check your val/loss graph.")

    def _calculate_validation_loss(self, tag, dataloader, get_model_prediction_and_target: Callable[
        [Any, Any], tuple[torch.Tensor, torch.Tensor]]) -> float:
        with torch.no_grad(), isolate_rng():
            # ok to override seed here because we are in a `with isolate_rng():` block
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            loss_validation_epoch = []
            steps_pbar = tqdm(range(len(dataloader)), position=1, leave=False)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Validate ({tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                loss_step = loss.detach().item()
                loss_validation_epoch.append(loss_step)

                steps_pbar.update(1)

            steps_pbar.close()

        loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)
        return loss_validation_local


    def _build_val_dataloader_if_required(self, image_train_items: list[ImageTrainItem], tokenizer)\
            -> tuple[Optional[torch.utils.data.DataLoader], list[ImageTrainItem]]:
        val_split_mode = self.config['val_split_mode'] if self.config['validate_training'] else None
        val_split_proportion = self.config['val_split_proportion']
        remaining_train_items = image_train_items
        if val_split_mode is None or val_split_mode == 'none':
            return None, image_train_items
        elif val_split_mode == 'automatic':
            val_items, remaining_train_items = get_random_split(image_train_items, val_split_proportion, batch_size=self.batch_size)
            val_items = list(disable_multiplier_and_flip(val_items))
            logging.info(f" * Removed {len(val_items)} images from the training set to use for validation")
        elif val_split_mode == 'manual':
            val_data_root = self.config.get('val_data_root', None)
            if val_data_root is None:
                raise ValueError("Manual validation split requested but `val_data_root` is not defined in validation config")
            val_items = self._load_manual_val_split(val_data_root)
            logging.info(f" * Loaded {len(val_items)} validation images from {val_data_root}")
        else:
            raise ValueError(f"Unrecognized validation split mode '{val_split_mode}'")
        val_ed_batch = self._build_ed_batch(val_items, batch_size=self.batch_size, tokenizer=tokenizer, name='val')
        val_dataloader = build_torch_dataloader(val_ed_batch, batch_size=self.batch_size)
        return val_dataloader, remaining_train_items

    def _build_train_stabilizer_dataloader_if_required(self, image_train_items: list[ImageTrainItem], tokenizer) \
            -> Optional[torch.utils.data.DataLoader]:
        stabilize_training_loss = self.config['stabilize_training_loss']
        if not stabilize_training_loss:
            return None

        stabilize_split_proportion = self.config['stabilize_split_proportion']
        stabilize_items, _ = get_random_split(image_train_items, stabilize_split_proportion, batch_size=self.batch_size)
        stabilize_items = list(disable_multiplier_and_flip(stabilize_items))
        stabilize_ed_batch = self._build_ed_batch(stabilize_items, batch_size=self.batch_size, tokenizer=tokenizer,
                                                  name='stabilize-train')
        stabilize_dataloader = build_torch_dataloader(stabilize_ed_batch, batch_size=self.batch_size)
        return stabilize_dataloader

    def _load_manual_val_split(self, val_data_root: str):
        args = Namespace(
            aspects=aspects.get_aspect_buckets(self.resolution),
            flip_p=0.0,
            seed=self.seed,
        )
        val_items = resolver.resolve_root(val_data_root, args)
        val_items.sort(key=lambda i: i.pathname)
        random.shuffle(val_items)
        return val_items

    def _build_ed_batch(self, items: list[ImageTrainItem], batch_size: int, tokenizer, name='val'):
        batch_size = self.batch_size
        seed = self.seed
        data_loader = DataLoaderMultiAspect(
            items,
            batch_size=batch_size,
            seed=seed,
        )
        ed_batch = EveryDreamBatch(
            data_loader=data_loader,
            debug_level=1,
            conditional_dropout=0,
            tokenizer=tokenizer,
            seed=seed,
            name=name,
            crop_jitter=0
        )
        return ed_batch
