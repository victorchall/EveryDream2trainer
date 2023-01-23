import json
import logging
import os
import random
import typing
import zipfile

import PIL.Image as Image
import tqdm
from colorama import Fore, Style

from data.image_train_item import ImageCaption, ImageTrainItem

class Event:
    def __init__(self, name: str):
        self.name = name
        
class UndersizedImageEvent(Event):
    def __init__(self, image_path: str, image_size: typing.Tuple[int, int], target_size: typing.Tuple[int, int]):
        super().__init__('undersized_image')
        self.image_path = image_path
        self.image_size = image_size
        self.target_size = target_size

class DataResolver:
    def __init__(self, aspects: list[typing.Tuple[int, int]], flip_p=0.0, seed=555):
        self.aspects = aspects
        self.flip_p = flip_p
        self.events = []
        
    def image_train_items(self, data_root: str) -> list[ImageTrainItem]:
        """
        Get the list of `ImageTrainItem` for the given data root.

        :param data_root: The data root, a directory, a file, etc..
        :return: The list of `ImageTrainItem`.
        """
        raise NotImplementedError()
    
    def compute_target_width_height(self, image_path: str) -> typing.Optional[typing.Tuple[int, int]]:
        # Compute the target width and height for the image based on the aspect ratio.
        with Image.open(image_path) as image:
            width, height = image.size
            image_aspect = width / height
            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            if width * height < target_wh[0] * target_wh[1]:
                event = UndersizedImageEvent(image_path, (width, height), target_wh)
                self.events.append(event)
            
            return target_wh
    
    def image_train_item(self, image_path: str, caption: ImageCaption, multiplier: float=1) -> ImageTrainItem:
        try:
            target_wh = self.compute_target_width_height(image_path)
            return ImageTrainItem(
                image=None,
                caption=caption,
                target_wh=target_wh,
                pathname=image_path,
                flip_p=self.flip_p,
                multiplier=multiplier
            )
        # TODO: This should only handle Image errors.
        except Exception as e:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{image_path}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {e}")
            

class JSONResolver(DataResolver):
    def image_train_items(self, json_path: str) -> list[ImageTrainItem]:
        items = []
        with open(json_path, encoding='utf-8', mode='r') as f:
            json_data = json.load(f)

        for data in json_data:
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
        skip_folders = []
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
                            logging.info(f" * DLMA multiply.txt in {current_dir} set to {val}")
                    except Exception as e:
                        logging.warning(f" * {Fore.LIGHTYELLOW_EX}Error trying to read multiply.txt for {current_dir}: {Style.RESET_ALL}{e}")
                        skip_folders.append(current_dir)
                        multipliers[current_dir] = 1.0
                else:
                    skip_folders.append(current_dir)
                    multipliers[current_dir] = 1.0
            
            caption = ImageCaption.resolve(pathname)
            item = self.image_train_item(pathname, caption, multiplier=multipliers[current_dir])
            
            cur_file_multiplier = multipliers[current_dir]

            while cur_file_multiplier >= 1.0:
                items.append(item)
                cur_file_multiplier -= 1
            
            if cur_file_multiplier > 0:
                if randomizer.random() < cur_file_multiplier:
                    items.append(item) 

            if item:
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

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            DirectoryResolver.__recurse_data_root(dir)

def strategy(data_root: str):
    if os.path.isfile(data_root) and data_root.endswith('.json'):
        return JSONResolver
    
    if os.path.isdir(data_root):
        return DirectoryResolver
        
    raise ValueError(f"data_root '{data_root}' is not a valid directory or JSON file.")
                    

def resolve_root(path: str, aspects: list[float], flip_p: float = 0.0, seed) -> typing.Tuple[list[ImageTrainItem], list[Event]]:
    """
    :param data_root: Directory or JSON file.
    :param aspects: The list of aspect ratios to use
    :param flip_p: The probability of flipping the image
    """
    if os.path.isfile(path) and path.endswith('.json'):
        resolver = JSONResolver(aspects, flip_p, seed)
    
    if os.path.isdir(path):
        resolver = DirectoryResolver(aspects, flip_p, seed)
        
    if not resolver:
        raise ValueError(f"data_root '{path}' is not a valid directory or JSON file.")

    items = resolver.image_train_items(path)
    events = resolver.events 
    return items, events

def resolve(value: typing.Union[dict, str], aspects: list[float], flip_p: float=0.0, seed=555) -> typing.Tuple[list[ImageTrainItem], list[Event]]:
    """
    Resolve the training data from the value.
    :param value: The value to resolve, either a dict or a string.
    :param aspects: The list of aspect ratios to use
    :param flip_p: The probability of flipping the image
    """
    if isinstance(value, str):
        return resolve_root(value, aspects, flip_p)
    
    if isinstance(value, dict):
        resolver = value.get('resolver', None)
        match resolver:
            case 'directory' | 'json':
                path = value.get('path', None)
                return resolve_root(path, aspects, flip_p, seed)
            case 'multi':
                resolved_items = []
                resolved_events = []
                for resolver in value.get('resolvers', []):
                    items, events = resolve(resolver, aspects, flip_p, seed)
                    resolved_items.extend(items)
                    resolved_events.extend(events)
                return resolved_items, resolved_events
            case _:
                raise ValueError(f"Cannot resolve training data for resolver value '{resolver}'")