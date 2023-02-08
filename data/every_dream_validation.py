import copy
import json
import logging
import math
import random
from typing import Callable, Any, Optional, Generator
from argparse import Namespace

import torch
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
    split_item_count = math.ceil(split_proportion * len(items) // batch_size) * batch_size
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
                 log_writer: SummaryWriter):
        self.val_dataloader = None
        self.train_overlapping_dataloader = None

        self.log_writer = log_writer
        self.resolution = resolution

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,

            'validate_training': True,
            'val_split_mode': 'automatic',
            'val_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15
        }
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                self.config.update(json.load(f))

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def every_n_epochs(self):
        return self.config['every_n_epochs']

    @property
    def seed(self):
        return self.config['seed']

    def prepare_validation_splits(self, train_items: list[ImageTrainItem], tokenizer: Any) -> list[ImageTrainItem]:
        """
        Build the validation splits as requested by the config passed at init.
        This may steal some items from `train_items`.
        If this happens, the returned `list` contains the remaining items after the required items have been stolen.
        Otherwise, the returned `list` is identical to the passed-in `train_items`.
        """
        with isolate_rng():
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
                self._do_validation('stabilize-train', global_step, self.train_overlapping_dataloader,
                                    get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable)

    def _do_validation(self, tag, global_step, dataloader, get_model_prediction_and_target: Callable[
        [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        with torch.no_grad(), isolate_rng():
            loss_validation_epoch = []
            steps_pbar = tqdm(range(len(dataloader)), position=1)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Validate ({tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                # ok to override seed here because we are in a `with isolate_rng():` block
                torch.manual_seed(self.seed + step)
                model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                loss_step = loss.detach().item()
                loss_validation_epoch.append(loss_step)

                steps_pbar.update(1)

            steps_pbar.close()

        loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)
        self.log_writer.add_scalar(tag=f"loss/{tag}", scalar_value=loss_validation_local, global_step=global_step)

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
            args = Namespace(
                aspects=aspects.get_aspect_buckets(self.resolution),
                flip_p=0.0,
                seed=self.seed,
            )
            val_data_root = self.config['val_data_root']
            val_items = resolver.resolve_root(val_data_root, args)
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
        )
        return ed_batch
