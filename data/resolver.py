import json
import logging
import os
import random
import typing
import zipfile
import argparse

import tqdm
from colorama import Fore, Style

from data.image_train_item import ImageCaption, ImageTrainItem

class DataResolver:
    def __init__(self, args: argparse.Namespace):
        """
        :param args: EveryDream configuration, an `argparse.Namespace` object.
        """
        self.aspects = args.aspects
        self.flip_p = args.flip_p
        self.seed = args.seed
        
    def image_train_items(self, data_root: str) -> list[ImageTrainItem]:
        """
        Get the list of `ImageTrainItem` for the given data root.

        :param data_root: The data root, a directory, a file, etc..
        :return: The list of `ImageTrainItem`.
        """
        raise NotImplementedError()
    
    def image_train_item(self, image_path: str, caption: ImageCaption, multiplier: float=1) -> ImageTrainItem:
        return ImageTrainItem(
            image=None,
            caption=caption,
            aspects=self.aspects,
            pathname=image_path,
            flip_p=self.flip_p,
            multiplier=multiplier
        )

class JSONResolver(DataResolver):
    def image_train_items(self, json_path: str) -> list[ImageTrainItem]:
        """
        Create `ImageTrainItem` objects with metadata for hydration later.
        Extracts images and captions from a JSON file.

        :param json_path: The path to the JSON file.
        """
        items = []
        with open(json_path, encoding='utf-8', mode='r') as f:
            json_data = json.load(f)

        for data in tqdm.tqdm(json_data):
            caption = JSONResolver.image_caption(data)
            if caption:
                image_value = JSONResolver.get_image_value(data)
                item = self.image_train_item(image_value, caption)
                if item:
                    items.append(item)

        return items
    
    @staticmethod
    def get_image_value(json_data: dict) -> typing.Optional[str]:
        """
        Get the image from the json data if possible.

        :param json_data: The json data, a dict.
        :return: The image, or None if not found.
        """
        image_value = json_data.get("image", None)
        if isinstance(image_value, str):
            image_value = image_value.strip()
            if os.path.exists(image_value):
                return image_value

    @staticmethod
    def get_caption_value(json_data: dict) -> typing.Optional[str]: 
        """
        Get the caption from the json data if possible.

        :param json_data: The json data, a dict.
        :return: The caption, or None if not found.
        """
        caption_value = json_data.get("caption", None)
        if isinstance(caption_value, str):
            return caption_value.strip()
    
    @staticmethod
    def image_caption(json_data: dict) -> typing.Optional[ImageCaption]:
        """
        Get the caption from the json data if possible.
        
        :param json_data: The json data, a dict.
        :return: The `ImageCaption`, or None if not found.
        """
        image_value = JSONResolver.get_image_value(json_data)
        caption_value = JSONResolver.get_caption_value(json_data)
        if image_value:
            if caption_value:
                return ImageCaption.resolve(caption_value)
            return ImageCaption.from_file(image_value)


class DirectoryResolver(DataResolver):    
    def image_train_items(self, data_root: str) -> list[ImageTrainItem]:
        """
        Create `ImageTrainItem` objects with metadata for hydration later.
        Unzips all zip files in `data_root` and then recursively searches the
        `data_root` for images and captions.

        :param data_root: The root directory to recurse through
        """
        DirectoryResolver.unzip_all(data_root)
        image_paths = list(DirectoryResolver.recurse_data_root(data_root))
        items = []
        multipliers = {}
        randomizer = random.Random(self.seed)
        
        for pathname in tqdm.tqdm(image_paths):
            current_dir = os.path.dirname(pathname)
            
            if current_dir not in multipliers:
                multiply_txt_path = os.path.join(current_dir, "multiply.txt")
                if os.path.exists(multiply_txt_path):
                    try:
                        with open(multiply_txt_path, 'r') as f:
                            val = float(f.read().strip())
                            multipliers[current_dir] = val
                            logging.info(f" - multiply.txt in '{current_dir}' set to {val}")
                    except Exception as e:
                        logging.warning(f" * {Fore.LIGHTYELLOW_EX}Error trying to read multiply.txt for {current_dir}: {Style.RESET_ALL}{e}")
                        multipliers[current_dir] = 1.0
                else:
                    multipliers[current_dir] = 1.0
            
            caption = ImageCaption.resolve(pathname)
            item = self.image_train_item(pathname, caption, multiplier=multipliers[current_dir])
            items.append(item)

        return items
        
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
    
    @staticmethod
    def recurse_data_root(recurse_root):
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']:
                    yield current

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                yield from DirectoryResolver.recurse_data_root(current)
        
def strategy(data_root: str) -> typing.Type[DataResolver]:
    """
    Determine the strategy to use for resolving the data.
    :param data_root: The root directory or JSON file to resolve.
    """
    if os.path.isfile(data_root) and data_root.endswith('.json'):
        return JSONResolver
    
    if os.path.isdir(data_root):
        return DirectoryResolver
        
    raise ValueError(f"data_root '{data_root}' is not a valid directory or JSON file.")
                    
def resolve_root(path: str, args: argparse.Namespace) -> list[ImageTrainItem]:
    """
    Resolve the training data from the root path.
    :param path: The root path to resolve.
    :param args: EveryDream configuration, an `argparse.Namespace` object.
    """
    resolver = strategy(path)
    return resolver(args).image_train_items(path)

def resolve(value: typing.Union[dict, str], args: argparse.Namespace) -> list[ImageTrainItem]:
    """
    Resolve the training data from the value.
    :param value: The value to resolve, either a dict, an array, or a string.
    :param args: EveryDream configuration, an `argparse.Namespace` object.
    """
    if isinstance(value, str):
        return resolve_root(value, args)
    
    if isinstance(value, dict):
        resolver = value.get('resolver', None)
        match resolver:
            case 'directory' | 'json':
                path = value.get('path', None)
                return resolve_root(path, args)
            case 'multi':
                return resolve(value.get('resolvers', []), args)
            case _:
                raise ValueError(f"Cannot resolve training data for resolver value '{resolver}'")

    if isinstance(value, list):
        items = []
        for item in value:
            items += resolve(item, args)
        return items 