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
import logging
import math
import os
import random
import typing
import yaml

import PIL
import PIL.Image as Image
import numpy as np
from torchvision import transforms

_RANDOM_TRIM = 0.04

DEFAULT_MAX_CAPTION_LENGTH = 2048

OptionalImageCaption = typing.Optional['ImageCaption']

class ImageCaption:
    """
    Represents the various parts of an image caption
    """

    def __init__(self, main_prompts: list[str], rating: float, tags: list[str], tag_weights: list[float], max_target_length: int, use_weights: bool):
        """
        :param main_prompt: The part of the caption which should always be included
        :param tags: list of tags to pick from to fill the caption
        :param tag_weights: weights to indicate which tags are more desired and should be picked preferably
        :param max_target_length: The desired maximum length of a generated caption
        :param use_weights: if ture, weights are considered when shuffling tags
        """
        self.__main_prompts = main_prompts
        self.__rating = rating
        self.__tags = tags
        self.__tag_weights = tag_weights
        self.__max_target_length = max_target_length
        self.__use_weights = use_weights
        if use_weights and len(tags) > len(tag_weights):
            self.__tag_weights.extend([1.0] * (len(tags) - len(tag_weights)))

        if use_weights and len(tag_weights) > len(tags):
            self.__tag_weights = tag_weights[:len(tags)]

    def rating(self) -> float:
        return self.__rating

    def get_shuffled_caption(self, seed: int) -> str:
        """
        returns the caption a string with a random selection of the tags in random order
        :param seed used to initialize the randomizer
        :return: generated caption string
        """
        rng = random.Random(seed)
        main_prompt = rng.choice(self.__main_prompts)
        if self.__tags:
            max_target_tag_length = self.__max_target_length - len(main_prompt)

            if self.__use_weights:
                tags_caption = self.__get_weighted_shuffled_tags(seed, self.__tags, self.__tag_weights, max_target_tag_length)
            else:
                tags_caption = self.__get_shuffled_tags(seed, self.__tags)

            return main_prompt + ", " + tags_caption
        return main_prompt

    def get_caption(self, seed) -> str:
        rng = random.Random(seed)
        main_prompt = rng.choice(self.__main_prompts)
        if self.__tags:            
            return main_prompt + ", " + ", ".join(self.__tags)
        return main_prompt

    @staticmethod
    def __get_weighted_shuffled_tags(seed: int, tags: list[str], weights: list[float], max_target_tag_length: int) -> str:
        picker = random.Random(seed)
        tags_copy = tags.copy()
        weights_copy = weights.copy()

        caption = ""
        while len(tags_copy) != 0 and len(caption) < max_target_tag_length:
            cum_weights = []
            weight_sum = 0.0
            for weight in weights_copy:
                weight_sum += weight
                cum_weights.append(weight_sum)

            point = picker.uniform(0, weight_sum)
            pos = bisect.bisect_left(cum_weights, point)

            weights_copy.pop(pos)
            tag = tags_copy.pop(pos)
            
            if caption:
                caption += ", "
            caption += tag

        return caption

    @staticmethod
    def __get_shuffled_tags(seed: int, tags: list[str]) -> str:
        random.Random(seed).shuffle(tags)
        return ", ".join(tags)

    @staticmethod
    def parse(lines: list[str]) -> 'ImageCaption':
        """
        Parses a string to get the caption.

        :param string: String to parse.
        :return: `ImageCaption` object.
        """
        main_prompts = []
        tags = []
        for line in lines:
            split_caption = list(map(str.strip, line.split(",")))
            main_prompts.append(split_caption[0])
            tags.extend(split_caption[1:])
        
        tag_weights = [1.0] * len(tags)

        return ImageCaption(main_prompts, 1.0, tags, tag_weights, DEFAULT_MAX_CAPTION_LENGTH, False)
    
    @staticmethod
    def from_file_name(file_path: str) -> 'ImageCaption':
        """
        Parses the file name to get the caption.
        
        :param file_path: Path to the image file.
        :return: `ImageCaption` object.
        """
        (file_name, _) = os.path.splitext(os.path.basename(file_path))
        caption = file_name.split("_")[0]
        return ImageCaption.parse(caption)
    
    @staticmethod
    def from_text_file(file_path: str, default_caption: OptionalImageCaption=None) -> OptionalImageCaption:
        """
        Parses a text file to get the caption. Returns the default caption if
        the file does not exist or is invalid.
        
        :param file_path: Path to the text file.
        :param default_caption: Optional `ImageCaption` to return if the file does not exist or is invalid.
        :return: `ImageCaption` object or `None`.
        """
        try:
            with open(file_path, encoding='utf-8', mode='r') as caption_file:
                caption_text = [line.rstrip() for line in caption_file]
                return ImageCaption.parse(caption_text)
        except:
            logging.error(f" *** Error reading {file_path} to get caption")
            return default_caption
        
    @staticmethod
    def from_yaml_file(file_path: str, default_caption: OptionalImageCaption=None) -> OptionalImageCaption:
        """
        Parses a yaml file to get the caption. Returns the default caption if
        the file does not exist or is invalid.
        
        :param file_path: path to the yaml file
        :param default_caption: caption to return if the file does not exist or is invalid
        :return: `ImageCaption` object or `None`.
        """
        try:
            with open(file_path, "r") as stream:
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
            logging.error(f" *** Error reading {file_path} to get caption")
            return default_caption
        
    @staticmethod
    def from_file(file_path: str, default_caption: OptionalImageCaption=None) -> OptionalImageCaption:
        """
        Try to resolve a caption from a file path or return `default_caption`.

        :string: The path to the file to parse.
        :default_caption: Optional `ImageCaption` to return if the file does not exist or is invalid.
        :return: `ImageCaption` object or `None`.
        """
        if os.path.exists(file_path):
            (file_path_without_ext, ext) = os.path.splitext(file_path) 
            match ext:
                case ".yaml" | ".yml":
                    return ImageCaption.from_yaml_file(file_path, default_caption)
                
                case ".txt" | ".caption":
                    return ImageCaption.from_text_file(file_path, default_caption)
                
                case '.jpg'| '.jpeg'| '.png'| '.bmp'| '.webp'| '.jfif':
                    for ext in [".yaml", ".yml", ".txt", ".caption"]:
                        file_path = file_path_without_ext + ext
                        image_caption = ImageCaption.from_file(file_path)
                        if image_caption is not None:
                            return image_caption
                    return ImageCaption.from_file_name(file_path)

                case _:
                    return default_caption
        else:
            return default_caption
        
    @staticmethod
    def resolve(string: str) -> 'ImageCaption':
        """
        Try to resolve a caption from a string. If the string is a file path,
        the caption will be read from the file, otherwise the string will be
        parsed as a caption.

        :string: The string to resolve.
        :return: `ImageCaption` object.
        """
        return ImageCaption.from_file(string, None) or ImageCaption.parse(string)


class ImageTrainItem:
    """
    image: PIL.Image
    identifier: caption,
    target_aspect: (width, height), 
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    rating: the relative rating of the images. The rating is measured in comparison to the other images.
    """
    def __init__(self, image: PIL.Image, caption: ImageCaption, aspects: list[float], pathname: str, flip_p=0.0, multiplier: float=1.0):
        self.caption = caption
        self.aspects = aspects
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.runt_size = 0
        self.multiplier = multiplier

        self.image_size = None
        if image is None or len(image) == 0:
            self.image = []
        else:
            self.image = image
            self.image_size = image.size
            self.target_size = None
            
        self.is_undersized = False
        self.error = None
        self.__compute_target_width_height()

    def hydrate(self, crop=False, crop_jitter=20):
        """
        crop: hard center crop to 512x512
        save: save the cropped image to disk, for manual inspection of resize/crop
        crop_jitter: randomly shift cropp by N pixels when using multiple aspect ratios to improve training quality
        """
        # print(self.pathname, self.image)
        try:
            # if not hasattr(self, 'image'):
            self.image = PIL.Image.open(self.pathname).convert('RGB')

            width, height = self.image.size
            if crop:
                cropped_img = self.__autocrop(self.image)
                self.image = cropped_img.resize((512, 512), resample=PIL.Image.BICUBIC)
            else:
                width, height = self.image.size
                jitter_amount = random.randint(0, crop_jitter)

                if self.target_wh[0] == self.target_wh[1]:
                    if width > height:
                        left = random.randint(0, width - height)
                        self.image = self.image.crop((left, 0, height + left, height))
                        width = height
                    elif height > width:
                        top = random.randint(0, height - width)
                        self.image = self.image.crop((0, top, width, width + top))
                        height = width
                    elif width > self.target_wh[0]:
                        slice = min(int(self.target_wh[0] * _RANDOM_TRIM), width - self.target_wh[0])
                        slicew_ratio = random.random()
                        left = int(slice * slicew_ratio)
                        right = width - int(slice * (1 - slicew_ratio))
                        sliceh_ratio = random.random()
                        top = int(slice * sliceh_ratio)
                        bottom = height - int(slice * (1 - sliceh_ratio))

                        self.image = self.image.crop((left, top, right, bottom))
                else:
                    image_aspect = width / height
                    target_aspect = self.target_wh[0] / self.target_wh[1]
                    if image_aspect > target_aspect:
                        new_width = int(height * target_aspect)
                        jitter_amount = max(min(jitter_amount, int(abs(width - new_width) / 2)), 0)
                        left = jitter_amount
                        right = left + new_width
                        self.image = self.image.crop((left, 0, right, height))
                    else:
                        new_height = int(width / target_aspect)
                        jitter_amount = max(min(jitter_amount, int(abs(height - new_height) / 2)), 0)
                        top = jitter_amount
                        bottom = top + new_height
                        self.image = self.image.crop((0, top, width, bottom))
                self.image = self.image.resize(self.target_wh, resample=PIL.Image.BICUBIC)

            self.image = self.flip(self.image)
        except Exception as e:
            logging.error(f"Fatal Error loading image: {self.pathname}:")
            logging.error(e)
            exit()

        if type(self.image) is not np.ndarray:
            self.image = np.array(self.image).astype(np.uint8)

        return self
    
    def __compute_target_width_height(self):
        self.target_wh = None
        try:
            with Image.open(self.pathname) as image:
                width, height = image.size
                image_aspect = width / height
                target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))
                
                self.is_undersized = (width * height) < (target_wh[0] * target_wh[1])
                self.target_wh = target_wh
        except Exception as e:
            self.error = e

    @staticmethod
    def __autocrop(image: PIL.Image, q=.404):
        """
        crops image to a random square inside small axis using a truncated gaussian distribution across the long axis
        """
        x, y = image.size

        if x != y:
            if (x > y):
                rand_x = x - y
                sigma = max(rand_x * q, 1)
            else:
                rand_y = y - x
                sigma = max(rand_y * q, 1)

            if (x > y):
                x_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = min(x_crop_gauss, (x - y) / 2)
                x_crop = math.trunc(x_crop)
                y_crop = 0
            else:
                y_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = 0
                y_crop = min(y_crop_gauss, (y - x) / 2)
                y_crop = math.trunc(y_crop)

            min_xy = min(x, y)
            image = image.crop((x_crop, y_crop, x_crop + min_xy, y_crop + min_xy))

        return image
