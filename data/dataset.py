import os
import logging
import yaml
import json

from functools import total_ordering
from attrs import define, Factory
from data.image_train_item import ImageCaption, ImageTrainItem
from utils.fs_helpers import *


@define(frozen=True)
@total_ordering
class Tag:
    value: str
    weight: float = None
    
    def __lt__(self, other):
        return self.value < other.value
    
@define(frozen=True)
@total_ordering
class Caption:
    main_prompt: str = None
    rating: float = None
    max_caption_length: int = None
    tags: frozenset[Tag] = Factory(frozenset)
    
    @classmethod
    def from_dict(cls, data: dict):
        main_prompt = data.get("main_prompt")
        rating = data.get("rating")
        max_caption_length = data.get("max_caption_length")

        tags = frozenset([ Tag(value=t.get("tag"), weight=t.get("weight")) 
                 for t in data.get("tags", [])
                 if "tag" in t and len(t.get("tag")) > 0])
        
        if not main_prompt and not tags:
            return None
        
        return Caption(main_prompt=main_prompt, rating=rating, max_caption_length=max_caption_length, tags=tags)

    @classmethod
    def from_text(cls, text: str):
        if text is None:
            return Caption(main_prompt="")
        split_caption = list(map(str.strip, text.split(",")))
        main_prompt = split_caption[0]
        tags = frozenset(Tag(value=t) for t in split_caption[1:])
        return Caption(main_prompt=main_prompt, tags=tags)
        
    @classmethod
    def load(cls, input):
        if isinstance(input, str):
            if os.path.isfile(input):
                return Caption.from_text(read_text(input))
            else:
                return Caption.from_text(input)
        elif isinstance(input, dict):
            return Caption.from_dict(input)
    
    def __lt__(self, other):
        self_str = ",".join([self.main_prompt] + sorted(repr(t) for t in self.tags))
        other_str = ",".join([other.main_prompt] + sorted(repr(t) for t in other.tags))
        return self_str < other_str

@define(frozen=True)
class ImageConfig:
    image: str = None
    captions: frozenset[Caption] = Factory(frozenset)
    multiply: float = None
    cond_dropout: float = None
    flip_p: float = None

    @classmethod
    def fold(cls, configs):
        acc = ImageConfig()
        [acc := acc.merge(cfg) for cfg in configs]
        return acc

    def merge(self, other):
        if other is None:
            return self

        if other.image and self.image:
            logging(f"Found two images with different extensions and the same barename: {self.image} and {other.image}")

        return ImageConfig(
            image = other.image or self.image,
            captions = other.captions.union(self.captions),
            multiply = other.multiply if other.multiply is not None else self.multiply,
            cond_dropout = other.cond_dropout if other.cond_dropout is not None else self.cond_dropout,
            flip_p = other.flip_p if other.flip_p is not None else self.flip_p
        )    

    def ensure_caption(self):
        if not self.captions:
            filename_caption = Caption.from_text(barename(self.image).split("_")[0])
            return self.merge(ImageConfig(captions=frozenset([filename_caption])))
        return self

    @classmethod
    def from_dict(cls, data: dict):
        captions = set()
        if "captions" in data:
            captions.update(Caption.load(cap) for cap in data.get("captions"))

        if "caption" in data:
            captions.add(Caption.load(data.get("caption")))
            
        if not captions:
            # For backward compatibility with existing caption yaml
            caption = Caption.load(data)
            if caption:
                captions.add(caption)
            
        return ImageConfig(
            image = data.get("image"),
            captions=frozenset(captions),
            multiply=data.get("multiply"),
            cond_dropout=data.get("cond_dropout"),
            flip_p=data.get("flip_p"))

    @classmethod
    def from_text(cls, text: str):
        try:
            if os.path.isfile(text):
                return ImageConfig.from_file(text)
            return ImageConfig(captions=frozenset({Caption.from_text(text)}))
        except Exception as e:
            logging.warning(f" *** Error parsing config from text {text}: \n{e}")

    @classmethod    
    def from_file(cls, file: str):
        try:
            match ext(file):
                case '.jpg' | '.jpeg' | '.png' | '.bmp' | '.webp' | '.jfif':
                    return ImageConfig(image=file)
                case ".json":
                    return ImageConfig.from_dict(json.load(read_text(file)))
                case ".yaml" | ".yml":
                    return ImageConfig.from_dict(yaml.safe_load(read_text(file)))
                case ".txt" | ".caption":
                    return ImageConfig.from_text(read_text(file))
                case _:
                    return logging.warning(" *** Unrecognized config extension {ext}")
        except Exception as e:
            logging.warning(f" *** Error parsing config from {file}: {e}")

    @classmethod
    def load(cls, input):
        if isinstance(input, str):
            return ImageConfig.from_text(input)
        elif isinstance(input, dict):
            return ImageConfig.from_dict(input)

@define()
class Dataset:
    image_configs: set[ImageConfig]

    def __global_cfg(files):
        cfgs = []
        for file in files:
            match os.path.basename(file):
                case 'global.yaml' | 'global.yml':
                    cfgs.append(ImageConfig.from_file(file))
        return ImageConfig.fold(cfgs)

    def __local_cfg(files):
        cfgs = []
        for file in files:
            match os.path.basename(file):
                case 'multiply.txt':
                    cfgs.append(ImageConfig(multiply=read_float(file)))
                case 'cond_dropout.txt':
                    cfgs.append(ImageConfig(cond_dropout=read_float(file)))
                case 'flip_p.txt':
                    cfgs.append(ImageConfig(flip_p=read_float(file)))
                case 'local.yaml' | 'local.yml':
                    cfgs.append(ImageConfig.from_file(file))
        return ImageConfig.fold(cfgs)

    def __image_cfg(imagepath, files):
        cfgs = [ImageConfig.from_file(imagepath)]
        for file in files:
            if same_barename(imagepath, file):
                match ext(file):
                    case '.txt' | '.caption' | '.yml' | '.yaml':
                        cfgs.append(ImageConfig.from_file(file))
        return ImageConfig.fold(cfgs)

    @classmethod
    def from_path(cls, data_root):
        # Create a visitor that maintains global config stack 
        # and accumulates image configs as it traverses dataset
        image_configs = set()
        def process_dir(files, parent_globals):
            globals = parent_globals.merge(Dataset.__global_cfg(files))
            locals = Dataset.__local_cfg(files)
            for img in filter(is_image, files):
                img_cfg = Dataset.__image_cfg(img, files)
                collapsed_cfg = ImageConfig.fold([globals, locals, img_cfg])
                resolved_cfg = collapsed_cfg.ensure_caption()
                image_configs.add(resolved_cfg)
            return globals

        walk_and_visit(data_root, process_dir, ImageConfig())
        return Dataset(image_configs)

    @classmethod
    def from_json(cls, json_path):
        """
        Import a dataset definition from a JSON file
        """
        configs = set()
        with open(json_path, encoding='utf-8', mode='r') as stream:
            for data in json.load(stream):
                cfg = ImageConfig.load(data).ensure_caption()
                if not cfg or not cfg.image:
                    logging.warning(f" *** Error parsing json image entry in {json_path}: {data}")
                    continue
                configs.add(cfg)
        return Dataset(configs)    
    
    def image_train_items(self, aspects):
        items = []
        for config in self.image_configs:
            caption = next(iter(sorted(config.captions)))
            if len(config.captions) > 1:
                logging.warning(f" *** Found multiple captions for image {config.image}, but only one will be applied: {config.captions}")

            use_weights = len(set(t.weight or 1.0 for t in caption.tags)) > 1
            tags = []
            tag_weights = []
            for tag in sorted(caption.tags):
                tags.append(tag.value)
                tag_weights.append(tag.weight or 1.0)
            use_weights = len(set(tag_weights)) > 1 
            
            caption = ImageCaption(
                main_prompt=caption.main_prompt,
                rating=caption.rating,
                tags=tags,
                tag_weights=tag_weights,
                max_target_length=caption.max_caption_length,
                use_weights=use_weights)

            item = ImageTrainItem(
                image=None,
                caption=caption,
                aspects=aspects,
                pathname=os.path.abspath(config.image),
                flip_p=config.flip_p or 0.0,
                multiplier=config.multiply or 1.0,
                cond_dropout=config.cond_dropout
            )
            items.append(item)
        return list(sorted(items, key=lambda ti: ti.pathname))


