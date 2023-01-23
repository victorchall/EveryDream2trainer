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
import bisect
import math
import os
import logging
import copy

import yaml
from PIL import Image
import random
from data.image_train_item import ImageTrainItem, ImageCaption
import data.aspects as aspects
from colorama import Fore, Style
import zipfile
import tqdm
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 715827880*4 # increase decompression bomb error limit to 4x default

DEFAULT_MAX_CAPTION_LENGTH = 2048

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
        self.has_scanned = False

        self.aspects = aspects.get_aspect_buckets(resolution=resolution, square_only=False)
        logging.info(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        logging.info(" Preloading images...")

        self.unzip_all(data_root)

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        self.prepared_train_data = self.__prescan_images(self.image_paths, flip_p)
        print(f"DLMA Loaded {len(self.prepared_train_data)} images")
        (self.rating_overall_sum, self.ratings_summed) = self.__sort_and_precalc_image_ratings()


    def __pick_multiplied_set(self, randomizer):
        """
        Deals with multiply.txt whole and fractional numbers
        """
        #print(f"Picking multiplied set from {len(self.prepared_train_data)}")
        data_copy = copy.deepcopy(self.prepared_train_data) # deep copy to avoid modifying original multiplier property
        epoch_size = len(self.prepared_train_data)
        picked_images = []

        # add by whole number part first and decrement multiplier in copy
        for iti in data_copy:
            #print(f"check for whole number {iti.multiplier}: {iti.pathname}, remaining {iti.multiplier}")
            while iti.multiplier >= 1.0:
                picked_images.append(iti)
                #print(f"Adding {iti.multiplier}: {iti.pathname}, remaining {iti.multiplier}, , datalen: {len(picked_images)}")
                iti.multiplier -= 1.0

        remaining = epoch_size - len(picked_images)

        assert remaining >= 0, "Something went wrong with the multiplier calculation"
        #print(f"Remaining to fill epoch after whole number adds: {remaining}")
        #print(f"Remaining in data copy: {len(data_copy)}")

        # add by renaming fractional numbers by random chance
        while remaining > 0:
            for iti in data_copy:
                if randomizer.uniform(0.0, 1.0) < iti.multiplier:
                    #print(f"Adding {iti.multiplier}: {iti.pathname}, remaining {remaining}, datalen: {len(data_copy)}")
                    picked_images.append(iti)
                    remaining -= 1
                    data_copy.remove(iti)
                if remaining <= 0:
                    break
        
        del data_copy
        return picked_images

    def get_shuffled_image_buckets(self, dropout_fraction: float = 1.0):
        """
        returns the current list of images including their captions in a randomized order,
        sorted into buckets with same sized images
        if dropout_fraction < 1.0, only a subset of the images will be returned
        if dropout_fraction >= 1.0, repicks fractional multipliers based on folder/multiply.txt values swept at prescan
        :param dropout_fraction: must be between 0.0 and 1.0.
        :return: randomized list of (image, caption) pairs, sorted into same sized buckets
        """

        self.seed += 1
        randomizer = random.Random(self.seed)

        if dropout_fraction < 1.0:
            picked_images = self.__pick_random_subset(dropout_fraction, randomizer)
        else:
            picked_images = self.__pick_multiplied_set(randomizer)

        randomizer.shuffle(picked_images)

        buckets = {}
        batch_size = self.batch_size
        for image_caption_pair in picked_images:
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
    def unzip_all(path):
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.zip'):
                        logging.info(f"Unzipping {file}")
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            zip_ref.extractall(path)
        except Exception as e:
            logging.error(f"Error unzipping files {e}")

    def __sort_and_precalc_image_ratings(self) -> tuple[float, list[float]]:
        self.prepared_train_data = sorted(self.prepared_train_data, key=lambda img: img.caption.rating())

        rating_overall_sum: float = 0.0
        ratings_summed: list[float] = []
        for image in self.prepared_train_data:
            rating_overall_sum += image.caption.rating()
            ratings_summed.append(rating_overall_sum)

        return rating_overall_sum, ratings_summed

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
    def __read_caption_from_yaml(file_path: str, fallback_caption: ImageCaption) -> ImageCaption:
        with open(file_path, "r") as stream:
            try:
                file_content = yaml.safe_load(stream)
                main_prompt = file_content.get("main_prompt", "")
                rating = file_content.get("rating", 1.0)
                unparsed_tags = file_content.get("tags", [])

                max_caption_length = file_content.get("max_caption_length", DEFAULT_MAX_CAPTION_LENGTH)

                tags = []
                tag_weights = []
                last_weight = None
                weights_differ = False
                for unparsed_tag in unparsed_tags:
                    tag = unparsed_tag.get("tag", "").strip()
                    if len(tag) == 0:
                        continue

                    tags.append(tag)
                    tag_weight = unparsed_tag.get("weight", 1.0)
                    tag_weights.append(tag_weight)

                    if last_weight is not None and weights_differ is False:
                        weights_differ = last_weight != tag_weight

                    last_weight = tag_weight

                return ImageCaption(main_prompt, rating, tags, tag_weights, max_caption_length, weights_differ)

            except:
                logging.error(f" *** Error reading {file_path} to get caption, falling back to filename")
                return fallback_caption

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

        return ImageCaption(main_prompt, 1.0, tags, [1.0] * len(tags), DEFAULT_MAX_CAPTION_LENGTH, False)

    def __prescan_images(self, image_paths: list, flip_p=0.0) -> list[ImageTrainItem]:
        """
        Create ImageTrainItem objects with metadata for hydration later
        """
        decorated_image_train_items = []

        if not self.has_scanned:
            undersized_images = []

        multipliers = {}
        skip_folders = []
        randomizer = random.Random(self.seed)

        for pathname in tqdm.tqdm(image_paths):
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]
            caption = DataLoaderMultiAspect.__split_caption_into_tags(caption_from_filename)

            file_path_without_ext = os.path.splitext(pathname)[0]
            yaml_file_path = file_path_without_ext + ".yaml"
            txt_file_path = file_path_without_ext + ".txt"
            caption_file_path = file_path_without_ext + ".caption"

            current_dir = os.path.dirname(pathname)

            try:
                if current_dir not in multipliers:
                    multiply_txt_path = os.path.join(current_dir, "multiply.txt")
                    #print(current_dir, multiply_txt_path)
                    if os.path.exists(multiply_txt_path):
                        with open(multiply_txt_path, 'r') as f:
                            val = float(f.read().strip())
                            multipliers[current_dir] = val
                            logging.info(f" * DLMA multiply.txt in {current_dir} set to {val}")
                    else:
                        skip_folders.append(current_dir)
                        multipliers[current_dir] = 1.0
            except Exception as e:
                logging.warning(f" * {Fore.LIGHTYELLOW_EX}Error trying to read multiply.txt for {current_dir}: {Style.RESET_ALL}{e}")
                skip_folders.append(current_dir)
                multipliers[current_dir] = 1.0

            if os.path.exists(yaml_file_path):
                caption = self.__read_caption_from_yaml(yaml_file_path, caption)
            elif os.path.exists(txt_file_path):
                caption = self.__read_caption_from_file(txt_file_path, caption)
            elif os.path.exists(caption_file_path):
                caption = self.__read_caption_from_file(caption_file_path, caption)

            try:
                image = Image.open(pathname)
                width, height = image.size
                image_aspect = width / height

                target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))
                if not self.has_scanned:
                    if width * height < target_wh[0] * target_wh[1]:
                        undersized_images.append(f" {pathname}, size: {width},{height}, target size: {target_wh}")

                image_train_item = ImageTrainItem(image=None, # image loaded at runtime to apply jitter
                                                  caption=caption, 
                                                  target_wh=target_wh, 
                                                  pathname=pathname, 
                                                  flip_p=flip_p,
                                                  multiplier=multipliers[current_dir],
                                                 )

                cur_file_multiplier = multipliers[current_dir]

                while cur_file_multiplier >= 1.0:
                    decorated_image_train_items.append(image_train_item)
                    cur_file_multiplier -= 1
                
                if cur_file_multiplier > 0:
                    if randomizer.random() < cur_file_multiplier:
                        decorated_image_train_items.append(image_train_item)
                
            except Exception as e:
                logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
                logging.error(f" *** exception: {e}")
                pass

        if not self.has_scanned:
            self.has_scanned = True
            if len(undersized_images) > 0:
                underized_log_path = os.path.join(self.log_folder, "undersized_images.txt")
                logging.warning(f"{Fore.LIGHTRED_EX} ** Some images are smaller than the target size, consider using larger images{Style.RESET_ALL}")
                logging.warning(f"{Fore.LIGHTRED_EX} ** Check {underized_log_path} for more information.{Style.RESET_ALL}")
                with open(underized_log_path, "w") as undersized_images_file:
                    undersized_images_file.write(f" The following images are smaller than the target size, consider removing or sourcing a larger copy:")
                    for undersized_image in undersized_images:
                        undersized_images_file.write(f"{undersized_image}\n")
        
        print (f" * DLMA: {len(decorated_image_train_items)} images loaded from {len(image_paths)} files")

        return decorated_image_train_items

    def __pick_random_subset(self, dropout_fraction: float, picker: random.Random) -> list[ImageTrainItem]:
        """
        Picks a random subset of all images
        - The size of the subset is limited by dropout_faction
        - The chance of an image to be picked is influenced by its rating. Double that rating -> double the chance
        :param dropout_fraction: must be between 0.0 and 1.0
        :param picker: seeded random picker
        :return: list of picked ImageTrainItem
        """

        prepared_train_data = self.prepared_train_data.copy()
        ratings_summed = self.ratings_summed.copy()
        rating_overall_sum = self.rating_overall_sum

        num_images = len(prepared_train_data)
        num_images_to_pick = math.ceil(num_images * dropout_fraction)
        num_images_to_pick = max(min(num_images_to_pick, num_images), 0)

        # logging.info(f"Picking {num_images_to_pick} images out of the {num_images} in the dataset for drop_fraction {dropout_fraction}")

        picked_images: list[ImageTrainItem] = []
        while num_images_to_pick > len(picked_images):
            # find random sample in dataset
            point = picker.uniform(0.0, rating_overall_sum)
            pos = min(bisect.bisect_left(ratings_summed, point), len(prepared_train_data) -1 )

            # pick random sample
            picked_image = prepared_train_data[pos]
            picked_images.append(picked_image)

            # kick picked item out of data set to not pick it again
            rating_overall_sum = max(rating_overall_sum - picked_image.caption.rating(), 0.0)
            ratings_summed.pop(pos)
            prepared_train_data.pop(pos)

        return picked_images

    @staticmethod
    def __recurse_data_root(self, recurse_root):
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']:
                    self.image_paths.append(current)

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)
