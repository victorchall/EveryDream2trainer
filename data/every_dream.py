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
import math
import data.dl_singleton as dls
from data.image_train_item import ImageTrainItem
import random
from torchvision import transforms
from transformers import CLIPTokenizer
import torch.nn.functional as F

class EveryDreamBatch(Dataset):
    """
    data_root: root path of all your training images, will be recursively searched for images
    repeats: how many times to repeat each image in the dataset
    flip_p: probability of flipping the image horizontally
    debug_level: 0=none, 1=print drops due to unfilled batches on aspect ratio buckets, 2=debug info per image, 3=save crops to disk for inspection
    batch_size: how many images to return in a batch
    conditional_dropout: probability of dropping the caption for a given image
    resolution: max resolution (relative to square)
    jitter: number of pixels to jitter the crop by, only for non-square images
    """
    def __init__(self,
                 data_root,
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1,
                 conditional_dropout=0.02,
                 resolution=512,
                 crop_jitter=20,
                 seed=555,
                 tokenizer=None,
                 log_folder=None,
                 ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        self.log_folder = log_folder
        #print(f"tokenizer: {tokenizer}")
        self.max_token_length = self.tokenizer.model_max_length

        if seed == -1:
            seed = random.randint(0, 99999)
        
        if not dls.shared_dataloader:
            logging.info(" * Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root,
                                         seed=seed,
                                         debug_level=debug_level,
                                         batch_size=self.batch_size,
                                         flip_p=flip_p,
                                         resolution=resolution,
                                         log_folder=self.log_folder,
                                        )
        
        self.image_train_items = dls.shared_dataloader.get_all_images()

        self.num_images = len(self.image_train_items)

        self._length = self.num_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        logging.info(f" ** Trainer Set: {self._length / batch_size:.0f}, num_images: {self.num_images}, batch_size: {self.batch_size}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        train_item = self.__get_image_for_trainer(self.image_train_items[i], self.debug_level)
        example["image"] = self.image_transforms(train_item["image"])

        #if random.random() > 9999:
        example["tokens"] = self.tokenizer(train_item["caption"],
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.tokenizer.model_max_length,
        ).input_ids
        example["tokens"] = torch.tensor(example["tokens"])
        example["caption"] = train_item["caption"] # for sampling if needed

        return example

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}

        save = debug_level > 2

        image_train_tmp = image_train_item.hydrate(crop=False, save=save, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image
        if random.random() > self.conditional_dropout:
            example["caption"] = image_train_tmp.caption
        else:
            example["caption"] = " "
        
        return example
