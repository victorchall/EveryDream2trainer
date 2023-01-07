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

import os
import logging
from PIL import Image
import random
from data.image_train_item import ImageTrainItem, ImageCaption
import data.aspects as aspects
from colorama import Fore, Style
import zipfile
import tqdm
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 715827880*4 # increase decompression bomb error limit to 4x default

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(self, data_root, seed=555, debug_level=0, batch_size=1, flip_p=0.0, resolution=512, log_folder=None):
        self.image_paths = []
        self.debug_level = debug_level
        self.flip_p = flip_p
        self.log_folder = log_folder
        self.seed = seed
        self.batch_size = batch_size
        self.runts = []

        self.aspects = aspects.get_aspect_buckets(resolution=resolution, square_only=False)
        logging.info(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        logging.info(" Preloading images...")

        self.unzip_all(data_root)

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        self.prepared_train_data = self.__prescan_images(self.image_paths, flip_p) # ImageTrainItem[]
        self.image_caption_pairs = self.__bucketize_images(self.prepared_train_data, batch_size=batch_size, debug_level=debug_level)
        
    def shuffle(self):
        self.runts = []
        self.seed = self.seed + 1
        random.Random(self.seed).shuffle(self.prepared_train_data)
        self.image_caption_pairs = self.__bucketize_images(self.prepared_train_data, batch_size=self.batch_size, debug_level=0)

    def unzip_all(self, path):
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.zip'):
                        logging.info(f"Unzipping {file}")
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            zip_ref.extractall(path)
        except Exception as e:
            logging.error(f"Error unzipping files {e}")

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __read_caption_from_file(file_path, fallback_caption: ImageCaption) -> ImageCaption:
        try:
            with open(file_path, encoding='utf-8', mode='r') as caption_file:
                caption_text = caption_file.read()
                caption = DataLoaderMultiAspect.__split_caption_into_tags(caption_text)
        except:
            logging.error(f" *** Error reading {file_path} to get caption, falling back to filename")
            caption = fallback_caption
            pass
        return caption

    @staticmethod
    def __split_caption_into_tags(caption_string: str) -> ImageCaption:
        """
        Splits a string by "," into the main prompt and additional tags with equal weights
        """
        split_caption = caption_string.split(",")
        main_prompt = split_caption.pop(0).strip()
        tags = []
        for tag in split_caption:
            tags.append(tag.strip())

        return ImageCaption(main_prompt, tags, [1.0] * len(tags))

    def __prescan_images(self, image_paths: list, flip_p=0.0):
        """
        Create ImageTrainItem objects with metadata for hydration later
        """
        decorated_image_train_items = []

        for pathname in tqdm.tqdm(image_paths):
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]
            caption = DataLoaderMultiAspect.__split_caption_into_tags(caption_from_filename)

            txt_file_path = os.path.splitext(pathname)[0] + ".txt"
            caption_file_path = os.path.splitext(pathname)[0] + ".caption"

            if os.path.exists(txt_file_path):
                caption = self.__read_caption_from_file(txt_file_path, caption)
            elif os.path.exists(caption_file_path):
                caption = self.__read_caption_from_file(caption_file_path, caption)

            try:
                image = Image.open(pathname)
                width, height = image.size
                image_aspect = width / height

                target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

                image_train_item = ImageTrainItem(image=None, caption=caption, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

                decorated_image_train_items.append(image_train_item)
            except Exception as e:
                logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
                logging.error(f" *** exception: {e}")
                pass

        return decorated_image_train_items

    def __bucketize_images(self, prepared_train_data: list, batch_size=1, debug_level=0):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            image_caption_pair.runt_size = 0
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair)

        if len(buckets) > 1:
            for bucket in buckets:
                truncate_count = len(buckets[bucket]) % batch_size
                if truncate_count > 0:
                    runt_bucket = buckets[bucket][-truncate_count:]
                    for item in runt_bucket:
                        item.runt_size = truncate_count
                    while len(runt_bucket) < batch_size:
                        runt_bucket.append(random.choice(runt_bucket))

                    current_bucket_size = len(buckets[bucket])

                    buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                    buckets[bucket].extend(runt_bucket)

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root):
        multiply = 1
        multiply_path = os.path.join(recurse_root, "multiply.txt")
        if os.path.exists(multiply_path):
            try: 
                with open(multiply_path, encoding='utf-8', mode='r') as f:
                    multiply = int(float(f.read().strip()))
                    logging.info(f" * DLMA multiply.txt in {recurse_root} set to {multiply}")
            except:
                logging.error(f" *** Error reading multiply.txt in {recurse_root}, defaulting to 1")
                pass

        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1]
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']:
                    # add image multiplyrepeats number of times
                    for _ in range(multiply):
                        self.image_paths.append(current)

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)
