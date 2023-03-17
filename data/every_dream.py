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
from data.data_loader import DataLoaderMultiAspect
from data.image_train_item import ImageTrainItem
import random
from torchvision import transforms
from transformers import CLIPTokenizer
import torch.nn.functional as F

class EveryDreamBatch(Dataset):
    """
    data_loader: `DataLoaderMultiAspect` object
    debug_level: 0=none, 1=print drops due to unfilled batches on aspect ratio buckets, 2=debug info per image, 3=save crops to disk for inspection
    conditional_dropout: probability of dropping the caption for a given image
    crop_jitter: number of pixels to jitter the crop by, only for non-square images
    seed: random seed
    """
    def __init__(self,
                 data_loader: DataLoaderMultiAspect,
                 debug_level=0,
                 conditional_dropout=0.02,
                 crop_jitter=20,
                 seed=555,
                 tokenizer=None,
                 retain_contrast=False,
                 shuffle_tags=False,
                 rated_dataset=False,
                 rated_dataset_dropout_target=0.5,
                 name='train',
                 attention_mask='none'
                 ):
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        self.max_token_length = self.tokenizer.model_max_length
        self.retain_contrast = retain_contrast
        self.shuffle_tags = shuffle_tags
        self.seed = seed
        self.rated_dataset = rated_dataset
        self.rated_dataset_dropout_target = rated_dataset_dropout_target
        # First epoch always trains on all images
        self.image_train_items  = []
        self.__update_image_train_items(1.0)
        self.name = name
        self.attention_mask = attention_mask
            
        num_images = len(self.image_train_items)
        logging.info(f" ** Dataset '{name}': {num_images / self.batch_size:.0f} batches, num_images: {num_images}, "
                     f"batch_size: {self.batch_size}, attention_mask: {self.attention_mask}")

    def shuffle(self, epoch_n: int, max_epochs: int):
        self.seed += 1
        
        if self.rated_dataset:
            dropout_fraction = (max_epochs - (epoch_n * self.rated_dataset_dropout_target)) / max_epochs
        else:
            dropout_fraction = 1.0
            
        self.__update_image_train_items(dropout_fraction)

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

        if random.random() > (train_item.get("cond_dropout", self.conditional_dropout)):
            tokens = self.tokenizer(example["caption"],
                                    truncation=True,
                                    padding="do_not_pad",
                                    max_length=self.tokenizer.model_max_length,
                                    ).input_ids
            # make mask and manually pad
            num_tokens_without_padding = len(tokens)
            padding_length = self.tokenizer.model_max_length - num_tokens_without_padding
            tokens += [self.tokenizer.pad_token_id] * padding_length
            if self.attention_mask == "none":
                attention_mask = [1] * self.tokenizer.model_max_length
            elif self.attention_mask == "pad":
                attention_mask = [1] * num_tokens_without_padding + [0] * padding_length
            elif self.attention_mask == "pad_and_bos_eos":
                attention_mask = [0] + [1] * (num_tokens_without_padding - 2) + [0] + [0] * padding_length
            else:
                raise ValueError(f"Unrecognized attention mask type: {self.attention_mask}")
        else:
            # conditional dropout is in effect
            tokens = self.tokenizer(" ",
                                    truncation=True,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    ).input_ids
            # train all those pad tokens too
            attention_mask = [1] * self.tokenizer.model_max_length


        example["tokens"] = torch.tensor(tokens)
        example["attention_mask"] = torch.tensor(attention_mask)

        example["runt_size"] = train_item["runt_size"]

        return example

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}
        save = debug_level > 2

        image_train_tmp = image_train_item.hydrate(crop=False, save=save, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image.copy()  # hack for now to avoid memory leak
        image_train_tmp.image = None # hack for now to avoid memory leak
        example["caption"] = image_train_tmp.caption
        if image_train_tmp.cond_dropout is not None:
            example["cond_dropout"] = image_train_tmp.cond_dropout
        example["runt_size"] = image_train_tmp.runt_size
       
        return example

    def __update_image_train_items(self, dropout_fraction: float):
        self.image_train_items = self.data_loader.get_shuffled_image_buckets(dropout_fraction)
        
def build_torch_dataloader(dataset, batch_size) -> torch.utils.data.DataLoader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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
    attention_mask = [example["attention_mask"] for example in batch]
    runt_size = batch[0]["runt_size"]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    ret = {
        "tokens": torch.stack(tuple(tokens)),
        "attention_mask": torch.stack(tuple(attention_mask)),
        "image": images,
        "captions": captions,
        "runt_size": runt_size,
    }
    del batch
    return ret
