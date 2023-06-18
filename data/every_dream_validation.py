import json
import logging
import math
import random
from dataclasses import dataclass, field
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

from colorama import Fore, Style


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


@dataclass
class ValidationDataset:
    name: str
    dataloader: torch.utils.data.DataLoader
    loss_history: list[float] = field(default_factory=list)
    val_loss_window_size: Optional[int] = 5  # todo: arg for this?

    def track_loss_trend(self, mean_loss: float):
        if self.val_loss_window_size is None:
            return
        self.loss_history.append(mean_loss)

        if len(self.loss_history) > ((self.val_loss_window_size * 2) + 1):
            dy = np.diff(self.loss_history[-self.val_loss_window_size:])
            if np.average(dy) > 0:
                logging.warning(f"Validation loss for {self.name} shows diverging.  Check your loss/{self.name} graph.")


class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 default_batch_size: int,
                 resolution: int,
                 log_writer: SummaryWriter,
    ):
        self.validation_datasets = []
        self.resolution = resolution
        self.log_writer = log_writer

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,

            'validate_training': True,
            'val_split_mode': 'automatic',
            'auto_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15,

            'use_relative_loss': False,

            'extra_manual_datasets': {
                # name: path pairs
                # eg "santa suit": "/path/to/captioned_santa_suit_images", will be logged to tensorboard as "loss/santa suit"
            }
        }
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                self.config.update(json.load(f))

        if 'val_data_root' in self.config:
            logging.warning(f"   * {Fore.YELLOW}using old name 'val_data_root' for 'manual_data_root' - please "
                  f"update your validation config json{Style.RESET_ALL}")
            self.config.update({'manual_data_root': self.config['val_data_root']})

        if self.config.get('val_split_mode') == 'manual':
            manual_data_root = self.config.get('manual_data_root')
            if manual_data_root is not None:
                self.config['extra_manual_datasets'].update({'val': self.config['manual_data_root']})
            else:
                if len(self.config['extra_manual_datasets']) == 0:
                    raise ValueError("Error in validation config .json: 'manual' validation requested but no "
                                     "'manual_data_root' or 'extra_manual_datasets'")

        if 'val_split_proportion' in self.config:
            logging.warning(f"   * {Fore.YELLOW}using old name 'val_split_proportion' for 'auto_split_proportion' - please "
                  f"update your validation config json{Style.RESET_ALL}")
            self.config.update({'auto_split_proportion': self.config['val_split_proportion']})



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

            auto_dataset, remaining_train_items = self._build_automatic_validation_dataset_if_required(train_items, tokenizer)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            train_overlapping_dataset = self._build_train_stabilizer_dataloader_if_required(
                remaining_train_items, tokenizer)

            if auto_dataset is not None:
                self.validation_datasets.append(auto_dataset)
            if train_overlapping_dataset is not None:
                self.validation_datasets.append(train_overlapping_dataset)
            manual_splits = self._build_manual_validation_datasets(tokenizer)
            self.validation_datasets.extend(manual_splits)

            return remaining_train_items

    def get_validation_step_indices(self, epoch, epoch_length_steps: int) -> list[int]:
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

    def do_validation(self, global_step: int,
                      get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        mean_loss_accumulator = 0
        for i, dataset in enumerate(self.validation_datasets):
            mean_loss = self._calculate_validation_loss(dataset.name,
                                                        dataset.dataloader,
                                                        get_model_prediction_and_target_callable)
            mean_loss_accumulator += mean_loss
            self.log_writer.add_scalar(tag=f"loss/{dataset.name}",
                                       scalar_value=mean_loss,
                                       global_step=global_step)
            dataset.track_loss_trend(mean_loss)
        # log combine loss to loss/_all_val_combined
        if len(self.validation_datasets) > 1:
            total_mean_loss = mean_loss_accumulator / len(self.validation_datasets)
            self.log_writer.add_scalar(tag=f"loss/_all_val_combined",
                                       scalar_value=total_mean_loss,
                                       global_step=global_step)

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


    def _build_automatic_validation_dataset_if_required(self, image_train_items: list[ImageTrainItem], tokenizer) \
            -> tuple[Optional[ValidationDataset], list[ImageTrainItem]]:
        val_split_mode = self.config['val_split_mode'] if self.config['validate_training'] else None
        if val_split_mode is None or val_split_mode == 'none' or val_split_mode == 'manual':
            # manual is handled by _build_manual_validation_datasets
            return None, image_train_items
        elif val_split_mode == 'automatic':
            auto_split_proportion = self.config['auto_split_proportion']
            val_items, remaining_train_items = get_random_split(image_train_items, auto_split_proportion, batch_size=self.batch_size)
            val_items = list(disable_multiplier_and_flip(val_items))
            logging.info(f" * Removed {len(val_items)} images from the training set to use for validation")
            val_ed_batch = self._build_ed_batch(val_items, tokenizer=tokenizer, name='val')
            val_dataloader = build_torch_dataloader(val_ed_batch, batch_size=self.batch_size)
            return ValidationDataset(name='val', dataloader=val_dataloader), remaining_train_items
        else:
            raise ValueError(f"Unrecognized validation split mode '{val_split_mode}'")

    def _build_manual_validation_datasets(self, tokenizer) -> list[ValidationDataset]:
        datasets = []
        for name, root in self.config.get('extra_manual_datasets', {}).items():
            items = self._load_manual_val_split(root)
            logging.info(f" * Loaded {len(items)} validation images for validation set '{name}' from {root}")
            ed_batch = self._build_ed_batch(items, tokenizer=tokenizer, name=name)
            dataloader = build_torch_dataloader(ed_batch, batch_size=self.batch_size)
            datasets.append(ValidationDataset(name=name, dataloader=dataloader))
        return datasets

    def _build_train_stabilizer_dataloader_if_required(self, image_train_items: list[ImageTrainItem], tokenizer) \
            -> Optional[ValidationDataset]:
        stabilize_training_loss = self.config['stabilize_training_loss']
        if not stabilize_training_loss:
            return None

        stabilize_split_proportion = self.config['stabilize_split_proportion']
        stabilize_items, _ = get_random_split(image_train_items, stabilize_split_proportion, batch_size=self.batch_size)
        stabilize_items = list(disable_multiplier_and_flip(stabilize_items))
        stabilize_ed_batch = self._build_ed_batch(stabilize_items, tokenizer=tokenizer, name='stabilize-train')
        stabilize_dataloader = build_torch_dataloader(stabilize_ed_batch, batch_size=self.batch_size)
        return ValidationDataset(name='stabilize-train', dataloader=stabilize_dataloader, val_loss_window_size=None)

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

    def _build_ed_batch(self, items: list[ImageTrainItem], tokenizer, name='val'):
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
