import json
import random
from typing import Callable, Any, Optional

import torch
from colorama import Fore, Style
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from utils.isolate_rng import isolate_rng


class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 train_batch: EveryDreamBatch,
                 log_writer: SummaryWriter):
        self.log_writer = log_writer

        val_config = {}
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                val_config = json.load(f)

        do_validation = val_config.get('validate_training', False)
        val_split_mode = val_config.get('val_split_mode', 'automatic') if do_validation else 'none'
        self.val_data_root = val_config.get('val_data_root', None)
        val_split_proportion = val_config.get('val_split_proportion', 0.15)

        stabilize_training_loss = val_config.get('stabilize_training_loss', False)
        stabilize_split_proportion = val_config.get('stabilize_split_proportion', 0.15)

        self.every_n_epochs = val_config.get('every_n_epochs', 1)
        self.seed = val_config.get('seed', 555)

        with isolate_rng():
            self.val_dataloader = self._build_validation_dataloader(val_split_mode,
                                                                    split_proportion=val_split_proportion,
                                                                    val_data_root=self.val_data_root,
                                                                    train_batch=train_batch)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            self.train_overlapping_dataloader = self._build_dataloader_from_automatic_split(train_batch,
                                                            split_proportion=stabilize_split_proportion,
                                                            name='train-stabilizer',
                                                            enforce_split=False) if stabilize_training_loss else None


    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                self._do_validation('stabilize-train', global_step, self.train_overlapping_dataloader, get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable)


    def _do_validation(self, tag, global_step, dataloader, get_model_prediction_and_target: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        with torch.no_grad(), isolate_rng():
            loss_validation_epoch = []
            steps_pbar = tqdm(range(len(dataloader)), position=1)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps ({tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                # ok to override seed here because we are in a `with isolate_rng():` block
                torch.manual_seed(self.seed + step)
                model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"])

                # del timesteps, encoder_hidden_states, noisy_latents
                # with autocast(enabled=args.amp):
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                loss_step = loss.detach().item()
                loss_validation_epoch.append(loss_step)

                steps_pbar.update(1)

            steps_pbar.close()

        loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)
        self.log_writer.add_scalar(tag=f"loss/{tag}", scalar_value=loss_validation_local, global_step=global_step)


    def _build_validation_dataloader(self,
                                     validation_split_mode: str,
                                     split_proportion: float,
                                     val_data_root: Optional[str],
                                     train_batch: EveryDreamBatch) -> Optional[DataLoader]:
        if validation_split_mode == 'none':
            return None
        elif validation_split_mode == 'automatic':
            return self._build_dataloader_from_automatic_split(train_batch, split_proportion, name='val', enforce_split=True)
        elif validation_split_mode == 'custom':
            if val_data_root is None:
                raise ValueError("val_data_root is required for 'split-custom' validation mode")
            return self._build_dataloader_from_custom_split(self.val_data_root, reference_train_batch=train_batch)
        else:
            raise ValueError(f"unhandled validation split mode '{validation_split_mode}'")


    def _build_dataloader_from_automatic_split(self,
                                               train_batch: EveryDreamBatch,
                                               split_proportion: float,
                                               name: str,
                                               enforce_split: bool=False) -> DataLoader:
        """
        Build a validation dataloader by copying data from the given `train_batch`. If `enforce_split` is `True`, remove
        the copied items from train_batch so that there is no overlap between `train_batch` and the new dataloader.
        """
        with isolate_rng():
            random.seed(self.seed)
            val_items = train_batch.get_random_split(split_proportion, remove_from_dataset=enforce_split)
            if enforce_split:
                print(
                f"  * Removed {len(val_items)} items for validation split from '{train_batch.name}' - {round(len(train_batch)/train_batch.batch_size)} batches are left")
            if len(train_batch) == 0:
                raise ValueError(f"Validation split used up all of the training data. Try a lower split proportion than {split_proportion}")
            val_batch = self._make_val_batch_with_train_batch_settings(
                val_items,
                train_batch,
                name=name
            )
            return build_torch_dataloader(
                items=val_batch,
                batch_size=train_batch.batch_size,
            )


    def _build_dataloader_from_custom_split(self, data_root: str, reference_train_batch: EveryDreamBatch) -> DataLoader:
        val_batch = self._make_val_batch_with_train_batch_settings(data_root, reference_train_batch)
        return build_torch_dataloader(
            items=val_batch,
            batch_size=reference_train_batch.batch_size
        )

    def _make_val_batch_with_train_batch_settings(self, data_root, reference_train_batch, name='val'):
        val_batch = EveryDreamBatch(
            data=data_root,
            debug_level=1,
            batch_size=reference_train_batch.batch_size,
            conditional_dropout=0,
            resolution=reference_train_batch.resolution,
            tokenizer=reference_train_batch.tokenizer,
            seed=reference_train_batch.seed,
            log_folder=reference_train_batch.log_folder,
            write_schedule=reference_train_batch.write_schedule,
            name=name,
        )
        return val_batch



