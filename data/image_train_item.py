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
import PIL.ImageOps as ImageOps
import numpy as np
from torchvision import transforms

_RANDOM_TRIM = 0.04


OptionalImageCaption = typing.Optional['ImageCaption']

class ImageCaption:
    """
    Represents the various parts of an image caption
    """
    def __init__(self, main_prompt: str, rating: float, tags: list[str], tag_weights: list[float], max_target_length: int, use_weights: bool):
        """
        :param main_prompt: The part of the caption which should always be included
        :param tags: list of tags to pick from to fill the caption
        :param tag_weights: weights to indicate which tags are more desired and should be picked preferably
        :param max_target_length: The desired maximum length of a generated caption
        :param use_weights: if ture, weights are considered when shuffling tags
        """
        self.__main_prompt = main_prompt
        self.__rating = rating
        self.__tags = tags
        self.__tag_weights = tag_weights
        self.__max_target_length = max_target_length or 2048
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
        if self.__tags:
            try:
                max_target_tag_length = self.__max_target_length - len(self.__main_prompt or 0)
            except Exception as e:
                print()
                logging.error(f"Error determining length for: {e} on {self.__main_prompt}")
                print()
                max_target_tag_length = 2048

            if self.__use_weights:
                tags_caption = self.__get_weighted_shuffled_tags(seed, self.__tags, self.__tag_weights, max_target_tag_length)
            else:
                tags_caption = self.__get_shuffled_tags(seed, self.__tags)

            return self.__main_prompt + ", " + tags_caption
        return self.__main_prompt

    def get_caption(self) -> str:
        if self.__tags:
            return self.__main_prompt + ", " + ", ".join(self.__tags)
        return self.__main_prompt

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

class ImageTrainItem:
    """
    image: PIL.Image
    identifier: caption,
    target_aspect: (width, height), 
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    rating: the relative rating of the images. The rating is measured in comparison to the other images.
    """
    def __init__(self, image: PIL.Image, caption: ImageCaption, aspects: list[float], pathname: str, flip_p=0.0, multiplier: float=1.0, cond_dropout=None):
        self.caption = caption
        self.aspects = aspects
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.runt_size = 0
        self.multiplier = multiplier
        self.cond_dropout = cond_dropout

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

    def load_image(self):
        image = PIL.Image.open(self.pathname).convert('RGB')
        try:
            image = ImageOps.exif_transpose(image)
        except SyntaxError as e:
            pass
        return image

    def hydrate(self, crop=False, save=False, crop_jitter=20):
        """
        crop: hard center crop to 512x512
        save: save the cropped image to disk, for manual inspection of resize/crop
        crop_jitter: randomly shift cropp by N pixels when using multiple aspect ratios to improve training quality
        """
        # print(self.pathname, self.image)
        try:
            # if not hasattr(self, 'image'):
            self.image = self.load_image()

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
            if save:
                base_name = os.path.basename(self.pathname)
                if not os.path.exists("test/output"):
                    os.makedirs("test/output")
                self.image.save(f"test/output/{base_name}")

            self.image = np.array(self.image).astype(np.uint8)

            # self.image = (self.image / 127.5 - 1.0).astype(np.float32)

        # print(self.image.shape)

        return self
    
    def __compute_target_width_height(self):
        self.target_wh = None
        try:
            with self.load_image() as image:
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
