"""
Copyright [2022-2024] Victor C Hall

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

import os
from typing import Generator
from data.image_train_item import ImageTrainItem, ImageCaption
from PIL import Image, ImageOps
import tarfile
import logging

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

class BucketBatchedGenerator(Generator[ImageTrainItem, None, None]):
    """
    returns items in with the same aspect ratio in batches, for use with batching dataloaders
    """
    def __init__(self, batch_size: int=1, generator: Generator[ImageTrainItem, None, None]=None):
        self.caption = batch_size
        self.cache = {}
        self.generator = generator

    def __iter__(self):
        for item in self.generator:
            if item.target_wh:
                aspect_bucket_key = item.target_wh
            if aspect_bucket_key not in self.cache:
                self.cache[aspect_bucket_key] = []
            self.cache[aspect_bucket_key].append(item)
            if len(self.cache[aspect_bucket_key]) >= self.batch_size:
                for item in self.cache[aspect_bucket_key]:
                    yield item
                self.cache[aspect_bucket_key] = []

# def image_train_item_generator_from_tar_pairs(image_dir: str, do_recurse: bool = True) -> Generator[ImageTrainItem, None, None]:
#     for root, dirs, files in os.walk(image_dir):
#         for file in files:
#             if file.endswith(".tar"):
#                 tar_path = os.path.join(root, file)
#                 with tarfile.open(tar_path, "r") as tar:
#                     for tarinfo in tar:
#                         if tarinfo.isfile() and any(tarinfo.name.endswith(ext) for ext in SUPPORTED_EXT):
#                             try:
#                                 img = Image.open(tar.extractfile(tarinfo))
#                                 txt = tar.extractfile(tarinfo.name.replace(os.path.splitext(tarinfo.name)[-1], ".txt"))
#                                 caption = txt.read().decode("utf-8")
#                                 img_caption = ImageCaption(main_prompt=caption, rating=0, tags=[], tag_weights=[], max_target_length=256, use_weights=False)
#                                 img = ImageOps.exif_transpose(img)
#                                 iti = ImageTrainItem(img, img_caption)
#                             except Exception as e:
#                                 logging.error(f"Failed to open {tarinfo.name}: {e}")
#                                 continue
#                             yield iti

def image_train_item_generator_from_files(image_dir: str, do_recurse: bool = True) -> Generator[ImageTrainItem, None, None]:
    for img_path in image_path_generator(image_dir, do_recurse):
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue
        # main_prompt: str, rating: float, tags: list[str], tag_weights: list[float], max_target_length: int, use_weights: bool):
        txt_cap_path = img_path.replace(os.path.splitext(img_path)[-1], ".txt")
        if os.path.exists(txt_cap_path):
            with open(txt_cap_path, "r") as f:
                caption = f.read()
        if not caption or len(caption) < 1:
            caption = os.path.basename(img_path)
            caption = caption.split("_")[0]
        image_caption = ImageCaption(main_prompt=caption, rating=0, tags=[], tag_weights=[], max_target_length=128, use_weights=False)
        iti = ImageTrainItem(img)
        yield iti

def image_path_generator(image_dir: str, do_recurse: bool = True) -> Generator[str, None, None]:
    if do_recurse:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                    yield os.path.join(root, file)
    else:
        for file in os.listdir(image_dir):
            if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                yield os.path.join(image_dir, file)