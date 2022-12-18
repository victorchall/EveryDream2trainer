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
from torch.utils.data import DataLoader
from data.every_dream import EveryDreamBatch

class EveryDreamDataLoaderWrapper(DataLoader):
    """
    Collates image:caption pairs into batches
    """
    def __init__(self, batch_size: int, tokenizer, dataset: EveryDreamBatch):
        self.dataset = dataset
        self.tokenizer = tokenizer

        super().__init__(dataset, batch_size, shuffle=False, pin_memory=True)
        #super().__init__(dataset, batch_size, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)

    def collate_fn(self, batch):
        """
        Collates batches of data
        based on https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        print("collate_fn")
        print(len(batch))
        captions = [example["caption"] for example in batch]
        images = [example["image"] for example in batch]

        print("collate_fn2")
        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()

        print("collate_fn3")
        captions = self.tokenizer.pad(
            {"captions": captions},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "captions": captions,
            "images": images,
        }
        print(f"{batch['captions']} {batch['images'].shape}")
        return batch