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
                 ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        #print(f"tokenizer: {tokenizer}")
        self.max_token_length = self.tokenizer.model_max_length

        if seed == -1:
            seed = random.randint(0, 99999)
        
        if not dls.shared_dataloader:
            print(" * Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root, seed=seed, debug_level=debug_level, batch_size=self.batch_size, flip_p=flip_p, resolution=resolution)
        
        self.image_train_items = dls.shared_dataloader.get_all_images()

        # for iti in self.image_train_items:
        #     print(f"iti caption:{iti.caption}")
        # exit()
        self.num_images = len(self.image_train_items)

        self._length = self.num_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print()
        print(f" ** Trainer Set: {self._length / batch_size:.0f}, num_images: {self.num_images}, batch_size: {self.batch_size}, length w/repeats: {self._length}")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        #print(" * Getting item", i)
        # batch = dict()
        # batch["images"] = list()
        # batch["captions"] = list()
        # first = True
        # for j in range(i, i + self.batch_size - 1):
        #     if j < self.num_images:
        #         example = self.__get_image_for_trainer(self.image_train_items[j], self.debug_level)
        #         if first:
        #             print(f"first example {j}", example)
        #             batch["images"] = [torch.from_numpy(example["image"])]
        #             batch["captions"] = [example["caption"]]
        #             first = False
        #         else: 
        #             print(f"subsiquent example {j}", example)
        #             batch["images"].extend(torch.from_numpy(example["image"]))
        #             batch["captions"].extend(example["caption"])
        example = {}

        train_item = self.__get_image_for_trainer(self.image_train_items[i], self.debug_level)
        #example["image"] = torch.from_numpy(train_item["image"])
        example["image"] = self.image_transforms(train_item["image"])
        # if train_item["caption"] == " ":
        #     example["tokens"] = [0 for i in range(self.max_token_length-2)]
        # else:
        if random.random() > self.conditional_dropout:
            example["tokens"] = self.tokenizer(train_item["caption"],
                                                #padding="max_length",
                                                truncation=True,
                                                padding=False,
                                                add_special_tokens=False,
                                                max_length=self.max_token_length-2,
            ).input_ids
            example["tokens"] = torch.tensor(example["tokens"])
        else:
            example["tokens"] = torch.zeros(75, dtype=torch.int)
        #print(f"bos: {self.tokenizer.bos_token_id}{self.tokenizer.eos_token_id}")

        #print(f"example['tokens']: {example['tokens']}")
        pad_amt = self.max_token_length-2 - len(example["tokens"])
        example['tokens']= F.pad(example['tokens'],pad=(0,pad_amt),mode='constant',value=0)
        example['tokens']= F.pad(example['tokens'],pad=(1,0),mode='constant',value=int(self.tokenizer.bos_token_id))
        eos_int = int(self.tokenizer.eos_token_id)
        #eos_int = int(0)
        example['tokens']= F.pad(example['tokens'],pad=(0,1),mode='constant',value=eos_int)
        #print(f"__getitem__ train_item['caption']: {train_item['caption']}")
        #print(f"__getitem__ train_item['pathname']: {train_item['pathname']}")
        #print(f"__getitem__ example['tokens'] pad: {example['tokens']}")

        example["caption"] = train_item["caption"] # for sampling if needed
        #print(f"len tokens: {len(example['tokens'])} cap: {example['caption']}")

        return example

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}

        save = debug_level > 2

        image_train_tmp = image_train_item.hydrate(crop=False, save=save, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image
        
        # if random.random() > self.conditional_dropout:
        example["caption"] = image_train_tmp.caption
        # else:
        #     example["caption"] = " "
        #print(f"      {image_train_tmp.pathname}: {image_train_tmp.caption}")
        return example
