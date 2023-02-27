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
                 shuffle_tags=False,
                 rated_dataset=False,
                 rated_dataset_dropout_target=0.5,
                 name='train'
                 ):
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        self.max_token_length = self.tokenizer.model_max_length
        self.shuffle_tags = shuffle_tags
        self.seed = seed
        self.rated_dataset = rated_dataset
        self.rated_dataset_dropout_target = rated_dataset_dropout_target
        # First epoch always trains on all images
        self.image_train_items  = []
        self.__update_image_train_items(1.0)
        self.name = name
            
        num_images = len(self.image_train_items)
        logging.info(f" ** Dataset '{name}': {num_images / self.batch_size:.0f} batches, num_images: {num_images}, batch_size: {self.batch_size}")

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

        train_item = self.__get_image_for_trainer(self.image_train_items[i])

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if self.shuffle_tags:
            example["caption"] = train_item["caption"].get_shuffled_caption(self.seed)
        else:
            example["caption"] = train_item["caption"].get_caption(self.seed)

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

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem):
        example = {}

        image_train_tmp = image_train_item.hydrate(crop=False, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image.copy()  # hack for now to avoid memory leak
        image_train_tmp.image = None # hack for now to avoid memory leak
        example["caption"] = image_train_tmp.caption
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
