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

import numpy as np
from torch.utils.data import Dataset
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
from ldm.data.image_train_item import ImageTrainItem

class EDValidateBatch(Dataset):
    def __init__(self,
                 data_root,
                 flip_p=0.0,
                 repeats=1,
                 debug_level=0,
                 batch_size=1,
                 set='val',
                 ):
        self.data_root = data_root
        self.batch_size = batch_size

        if not dls.shared_dataloader:
            print("Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size, flip_p=flip_p)
            
        self.image_train_items = dls.shared_dataloader.get_all_images()
        
        self.num_images = len(self.image_train_items)

        self._length = max(math.trunc(self.num_images * repeats), batch_size) - self.num_images % self.batch_size

        print()
        print(f" ** Validation Set: {set}, steps: {self._length / batch_size:.0f}, repeats: {repeats} ")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        image_train_item = self.image_train_items[idx]

        example = self.__get_image_for_trainer(image_train_item)
        return example

    @staticmethod
    def __get_image_for_trainer(image_train_item: ImageTrainItem):
        example = {}

        image_train_tmp = image_train_item.hydrate()

        example["image"] = image_train_tmp.image
        example["caption"] = image_train_tmp.caption

        return example
        