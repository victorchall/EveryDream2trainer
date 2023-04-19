import cProfile
from contextlib import nullcontext
import os
import logging
import time
import yaml
import json

from functools import partial
from attrs import define, field
from data.image_train_item import ImageCaption, ImageTrainItem
from utils.fs_helpers import *
from typing import Iterable

from tqdm import tqdm

from multiprocessing import Pool, Lock

DEFAULT_MAX_CAPTION_LENGTH = 2048

def overlay(overlay, base):
    return overlay if overlay is not None else base

def safe_set(val):
    if isinstance(val, str):
        return dict.fromkeys([val]) if val else dict()

    if isinstance(val, Iterable):
        return dict.fromkeys((i for i in val if i is not None))
    
    return val or dict() 

@define(frozen=True)
class Tag:
    value: str
    weight: float = field(default=1.0, converter=lambda x: x if x is not None else 1.0)

    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            return Tag(data)

        if isinstance(data, dict):
            value = data.get("tag")
            weight = data.get("weight")
            if value:
                return Tag(value, weight)

        return None

@define
class ImageConfig:
    # Captions
    main_prompts: dict[str, None] = field(factory=dict, converter=safe_set)
    rating: float = None
    max_caption_length: int = None
    tags: dict[Tag, None] = field(factory=dict, converter=safe_set)
    
    # Options
    multiply: float = None
    cond_dropout: float = None
    flip_p: float = None

    def merge(self, other):
        if other is None:
            return self

        return ImageConfig(
            main_prompts=other.main_prompts | self.main_prompts,
            rating=overlay(other.rating, self.rating),
            max_caption_length=overlay(other.max_caption_length, self.max_caption_length),
            tags= other.tags | self.tags,
            multiply=overlay(other.multiply, self.multiply),
            cond_dropout=overlay(other.cond_dropout, self.cond_dropout),
            flip_p=overlay(other.flip_p, self.flip_p),
        )

    @classmethod
    def from_dict(cls, data: dict):
        # Parse standard yaml tag file (with options)
        parsed_cfg = ImageConfig(
            main_prompts=safe_set(data.get("main_prompt")), 
            rating=data.get("rating"), 
            max_caption_length=data.get("max_caption_length"), 
            tags=safe_set(map(Tag.parse, data.get("tags", []))),
            multiply=data.get("multiply"),
            cond_dropout=data.get("cond_dropout"),
            flip_p=data.get("flip_p"),
            )

        # Alternatively parse from dedicated `caption` attribute
        if cap_attr := data.get('caption'):
            parsed_cfg = parsed_cfg.merge(ImageConfig.parse(cap_attr))

        return parsed_cfg

    @classmethod
    def fold(cls, configs):
        acc = ImageConfig()
        for cfg in configs:
            acc = acc.merge(cfg)
        return acc

    def ensure_caption(self):
        return self

    @classmethod
    def from_caption_text(cls, text: str):
        if not text:
            return ImageConfig()
        if os.path.isfile(text):
            return ImageConfig.from_file(text)

        split_caption = list(map(str.strip, text.split(",")))
        return ImageConfig(
            main_prompts=split_caption[0], 
            tags=map(Tag.parse, split_caption[1:])
            )

    @classmethod    
    def from_file(cls, file: str):
        match ext(file):
            case '.jpg' | '.jpeg' | '.png' | '.bmp' | '.webp' | '.jfif':
                return ImageConfig(image=file)
            case ".json":
                return ImageConfig.from_dict(json.load(read_text(file)))
            case ".yaml" | ".yml":
                return ImageConfig.from_dict(yaml.safe_load(read_text(file)))
            case ".txt" | ".caption":
                return ImageConfig.from_caption_text(read_text(file))
            case _:
                return logging.warning(" *** Unrecognized config extension {ext}")

    @classmethod
    def parse(cls, input):
        if isinstance(input, str):
            if os.path.isfile(input):
                return ImageConfig.from_file(input)
            else:
                return ImageConfig.from_caption_text(input)
        elif isinstance(input, dict):
            return ImageConfig.from_dict(input)
    

@define()
class Dataset:
    image_configs: dict[str, ImageConfig]

    def __global_cfg(fileset):
        cfgs = []
        
        for cfgfile in ['global.yaml', 'global.yml']:
            if cfgfile in fileset:
                cfgs.append(ImageConfig.from_file(fileset[cfgfile]))
        return ImageConfig.fold(cfgs)

    def __local_cfg(fileset):
        cfgs = []
        if 'multiply.txt' in fileset:
            cfgs.append(ImageConfig(multiply=read_float(fileset['multiply.txt'])))
        if 'cond_dropout.txt' in fileset:
            cfgs.append(ImageConfig(cond_dropout=read_float(fileset['cond_dropout.txt'])))
        if 'flip_p.txt' in fileset:
            cfgs.append(ImageConfig(flip_p=read_float(fileset['flip_p.txt'])))
        if 'local.yaml' in fileset:
            cfgs.append(ImageConfig.from_file(fileset['local.yaml']))
        if 'local.yml' in fileset:
            cfgs.append(ImageConfig.from_file(fileset['local.yml']))
        return ImageConfig.fold(cfgs)

    def __sidecar_cfg(imagepath, fileset, lock):
        cfgs = []
        for cfgext in ['.txt', '.caption', '.yml', '.yaml']:
            cfgfile = barename(imagepath) + cfgext
            if cfgfile in fileset:
                cfg = ImageConfig.from_file(fileset[cfgfile])
                with lock:
                    cfgs.append(cfg)
        return ImageConfig.fold(cfgs)

    # Use file name for caption only as a last resort
    @classmethod
    def __ensure_caption(cls, cfg: ImageConfig, file: str):
        if cfg.main_prompts:
            return cfg
        cap_cfg = ImageConfig.from_caption_text(barename(file).split("_")[0])
        return cfg.merge(cap_cfg)

    @classmethod
    def scan_one(cls, img, image_configs, fileset, global_cfg, local_cfg, lock):
        img_cfg = Dataset.__sidecar_cfg(img, fileset, lock)
        resolved_cfg = ImageConfig.fold([global_cfg, local_cfg, img_cfg])
        with lock:
            image_configs[img] = Dataset.__ensure_caption(resolved_cfg, img)

    @classmethod
    def scan_one_full(cls, img, image_configs, fileset, global_cfg, local_cfg, lock):
        Dataset.scan_one(img, image_configs, fileset, global_cfg, local_cfg, lock)
        img_cfg = Dataset.__sidecar_cfg(img, fileset, lock)
        resolved_cfg = ImageConfig.fold([global_cfg, local_cfg, img_cfg])
        image_configs[img] = Dataset.__ensure_caption(resolved_cfg, img)
        #print(f"{image_configs[img].main_prompts} {image_configs[img].tags} {image_configs[img].rating}")


    @classmethod
    def from_path(cls, data_root):
        # Create a visitor that maintains global config stack 
        # and accumulates image configs as it traverses dataset

        image_configs = {}
        def process_dir(files, parent_globals):
            #pool = Pool(int(os.cpu_count()/2))
            lock = Lock()

            fileset = {os.path.basename(f): f for f in files}
            global_cfg = parent_globals.merge(Dataset.__global_cfg(fileset))
            local_cfg = Dataset.__local_cfg(fileset)
            for img in filter(is_image, files):
                #pool.apply_async(Dataset.scan_one_full, args=(img, image_configs, fileset, global_cfg, local_cfg, lock))
                Dataset.scan_one_full(img, image_configs, fileset, global_cfg, local_cfg, lock)
                #Dataset.scan_one(img, image_configs, fileset, global_cfg, local_cfg, lock)
            #pool.close()
            #pool.join()
            #     img_cfg = Dataset.__sidecar_cfg(img, fileset)
            #     resolved_cfg = ImageConfig.fold([global_cfg, local_cfg, img_cfg])
            #     image_configs[img] = Dataset.__ensure_caption(resolved_cfg, img)
            
            return global_cfg

        time_start = time.time()
        walk_and_visit(data_root, process_dir, ImageConfig())
        time_end = time.time()
        logging.info(f"   ... walk_and_visit took {(time_end - time_start)/60:.2f} minutes and found {len(image_configs)} images")

        return Dataset(image_configs)

    @classmethod
    def from_json(cls, json_path):
        """
        Import a dataset definition from a JSON file
        """
        image_configs = {}
        with open(json_path, encoding='utf-8', mode='r') as stream:
            for data in json.load(stream):
                img = data.get("image")
                cfg = Dataset.__ensure_caption(ImageConfig.parse(data), img)
                if not img:
                    logging.warning(f" *** Error parsing json image entry in {json_path}: {data}")
                    continue
                image_configs[img] = cfg
        return Dataset(image_configs)    

    def get_one_image_train_item(self, image, aspects, profile=False) -> ImageTrainItem:
        
        
        config = self.image_configs[image]

        tags = []
        tag_weights = []
        for tag in sorted(config.tags, key=lambda x: x.weight or 1.0, reverse=True):
            tags.append(tag.value)
            tag_weights.append(tag.weight)
        use_weights = len(set(tag_weights)) > 1 

        try:
            if profile:
                profiler = cProfile.Profile()
                import random
                random_n = f"{random.randint(0,999):03d}"
                profiler.enable()
            caption = ImageCaption(
                main_prompt=next(iter(config.main_prompts)),
                rating=config.rating or 1.0,
                tags=tags,
                tag_weights=tag_weights,
                max_target_length=config.max_caption_length or DEFAULT_MAX_CAPTION_LENGTH,
                use_weights=use_weights)
            if profile:
                profiler.disable()
                profiler.dump_stats(f'profile{random_n}.prof')
                #exit()

            item = ImageTrainItem(
                image=None,
                caption=caption,
                aspects=aspects,
                pathname=os.path.abspath(image),
                flip_p=config.flip_p or 0.0,
                multiplier=config.multiply or 1.0,
                cond_dropout=config.cond_dropout
            )
        except Exception as e:
            logging.error(f" *** Error preloading image or caption for: {image}, error: {e}")
            raise e

        
        return item

    def image_train_items(self, aspects):
        print(f"   * using async loader")
        run_profiler = False
        items = []
        process_count = int(os.cpu_count()/2)
        pool = Pool(process_count)
        async_results = []

        time_start = time.time()
        with tqdm(total=len(self.image_configs), desc=f"preloading {process_count}", dynamic_ncols=True) as pbar:
                for image in self.image_configs:
                    async_result = pool.apply_async(self.get_one_image_train_item, args=(image,aspects, run_profiler), callback=lambda _: pbar.update())
                    async_results.append(async_result)
                pool.close()
                pool.join()

                for async_result in async_results:
                    result = async_result.get()
                    if result is not None:
                        # ImageTrainItem
                        items.append(result)
                    else:
                        raise ValueError(" *** image_train_items(): Async load item missing")
                
                
        
        time_end = time.time()
        logging.info(f" *** Preloading took {(time_end - time_start)/60:.2f} minutes and found {len(items)} images")
        return items

    def image_train_items_newish(self, aspects):
        print(f"   * using async loader")
        items = []
        process_count = int(os.cpu_count()/2)
        pool = Pool(process_count)

        time_start = time.time()
        with tqdm(total=len(self.image_configs), desc=f"preloading {process_count}", dynamic_ncols=True) as pbar:
            async_results = []
           
            # run 1000 async tasks
            for image in self.image_configs:
                # profile the task
                #cProfile.runctx('self.get_one(image,aspects)', globals(), locals(), 'profile.prof')
                async_result = pool.apply_async(self.get_one_image_train_item, args=(image,aspects), callback=lambda _: pbar.update())
                async_results.append(async_result)
            pool.close()
            #pool.join()
            print(f"   * async pool closed")

            for async_result in async_results:
                result = async_result.get()
                if result is not None:
                    # ImageTrainItem
                    items.append(result)
                    print(f"{result.pathname} {result.caption.main_prompt}")
                else:
                    raise ValueError(" *** image_train_items(): Async load item missing")
        
        time_end = time.time()
        logging.info(f" *** Preloading took {(time_end - time_start)/60:.2f} minutes and found {len(items)} images")
        return items

    def image_train_items_old(self, aspects):
        print(f"   * using single threaded loader")
        items = []

        time_start = time.time()
        with tqdm(total=len(self.image_configs), desc="preloading", dynamic_ncols=True) as pbar:
            for image in self.image_configs:
                items.append(self.get_one_image_train_item(image, aspects))
                pbar.update()
        time_end = time.time()
        logging.info(f" *** Preloading took {(time_end - time_start)/60:.2f} minutes and found {len(items)} images")
        return items
