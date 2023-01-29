"""
Copyright [2022] Victor C Hall

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

import torch
from torch.utils.data import Dataset
from data.data_loader import DataLoaderMultiAspect as dlma
from data.image_train_item import ImageTrainItem
import random
from torchvision import transforms


class EveryDreamBatch(Dataset):
    """
    data: either a str indicating the root path of all your training images, will be recursively searched for images; or a list of ImageTrainItem
    repeats: how many times to repeat each image in the dataset
    flip_p: probability of flipping the image horizontally
    debug_level: 0=none, 1=print drops due to unfilled batches on aspect ratio buckets, 2=debug info per image, 3=save crops to disk for inspection
    batch_size: how many images to return in a batch
    conditional_dropout: probability of dropping the caption for a given image
    resolution: max resolution (relative to square)
    jitter: number of pixels to jitter the crop by, only for non-square images
    name: the name of this dataset (only used for logging)
    """
    def __init__(self,
                 data: str | list[ImageTrainItem],
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1,
                 conditional_dropout=0.02,
                 resolution=512,
                 crop_jitter=20,
                 seed=555,
                 tokenizer=None,
                 log_folder=None,
                 retain_contrast=False,
                 write_schedule=False,
                 shuffle_tags=False,
                 rated_dataset=False,
                 rated_dataset_dropout_target=0.5,
                 name='train'
                 ):
        self.batch_size = batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.resolution = resolution
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        self.log_folder = log_folder
        #print(f"tokenizer: {tokenizer}")
        self.max_token_length = self.tokenizer.model_max_length
        self.retain_contrast = retain_contrast
        self.write_schedule = write_schedule
        self.shuffle_tags = shuffle_tags
        self.seed = seed
        self.rated_dataset = rated_dataset
        self.rated_dataset_dropout_target = rated_dataset_dropout_target
        self.name = name

        if seed == -1:
            seed = random.randint(0, 99999)

        self.dataloader = dlma(data=data,
                               seed=seed,
                               debug_level=debug_level,
                               batch_size=self.batch_size,
                               flip_p=flip_p,
                               resolution=resolution,
                               log_folder=self.log_folder,
                               name=self.name
                               )
        self.__update_image_train_items(1.0, 0)

        num_images = len(self.image_train_items)
        logging.info(f" ** EveryDreamBatch Set '{self.name}': {num_images / batch_size:.0f} batches, num_images: {num_images}, batch_size: {self.batch_size}")


    def get_random_split(self, split_proportion: float, remove_from_dataset: bool=False) -> list[ImageTrainItem]:
        if self.dataloader is None:
            raise RuntimeError("EveryDreamBatch is already static")
        items = self.dataloader.get_random_split(split_proportion, remove_from_dataset=remove_from_dataset)
        self.__update_image_train_items(1.0, 0)
        return items


    def shuffle(self, epoch_n: int, max_epochs: int):
        self.seed += 1
        if self.rated_dataset:
            dropout_fraction = (max_epochs - (epoch_n * self.rated_dataset_dropout_target)) / max_epochs
        else:
            dropout_fraction = 1.0
        self.__update_image_train_items(dropout_fraction, epoch_n)


    def __update_image_train_items(self, dropout_fraction: float, epoch_n: int):
        if self.dataloader is None:
            raise RuntimeError("Cannot run __update_train_images on a static EveryDreamBatch")
        self.image_train_items = self.dataloader.get_shuffled_image_buckets(dropout_fraction)
        if self.write_schedule:
            self.__write_batch_schedule(epoch_n + 1)


    def __write_batch_schedule(self, epoch_n):
        with open(f"{self.log_folder}/ep{epoch_n}_batch_schedule_{self.name}.txt", "w", encoding='utf-8') as f:
            for i in range(len(self.image_train_items)):
                try:
                    f.write(f"step:{int(i / self.batch_size):05}, wh:{self.image_train_items[i].target_wh}, r:{self.image_train_items[i].runt_size}, path:{self.image_train_items[i].pathname}\n")
                except Exception as e:
                    logging.error(f" * Error writing to batch schedule for file path: {self.image_train_items[i].pathname}")


    def __len__(self):
        return len(self.image_train_items)


    def __getitem__(self, i):
        example = {}

        train_item = self.__get_image_for_trainer(self.image_train_items[i], self.debug_level)

        if self.retain_contrast:
            std_dev = 1.0
            mean = 0.0
        else:
            std_dev = 0.5
            mean = 0.5

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([mean], [std_dev]),
            ]
        )

        if self.shuffle_tags:
            example["caption"] = train_item["caption"].get_shuffled_caption(self.seed)
        else:
            example["caption"] = train_item["caption"].get_caption()

        example["image"] = image_transforms(train_item["image"])

        if random.random() > self.conditional_dropout:
            example["tokens"] = self.tokenizer(example["caption"],
                                                truncation=True,
                                                padding="max_length",
                                                max_length=self.tokenizer.model_max_length,
                                              ).input_ids
        else:
            example["tokens"] = self.tokenizer(" ",
                                                truncation=True,
                                                padding="max_length",
                                                max_length=self.tokenizer.model_max_length,
                                              ).input_ids

        example["tokens"] = torch.tensor(example["tokens"])

        example["runt_size"] = train_item["runt_size"]

        return example


    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}
        save = debug_level > 2

        image_train_tmp = image_train_item.hydrate(crop=False, save=save, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image
        example["caption"] = image_train_tmp.caption
        example["runt_size"] = image_train_tmp.runt_size
        return example


def build_torch_dataloader(items, batch_size) -> torch.utils.data.DataLoader:
    dataloader = torch.utils.data.DataLoader(
        items,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    """
    Collates batches
    """
    images = [example["image"] for example in batch]
    captions = [example["caption"] for example in batch]
    tokens = [example["tokens"] for example in batch]
    runt_size = batch[0]["runt_size"]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    ret = {
        "tokens": torch.stack(tuple(tokens)),
        "image": images,
        "captions": captions,
        "runt_size": runt_size,
    }
    del batch
    return ret


