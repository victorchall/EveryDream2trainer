import json
import logging
import os.path
from dataclasses import dataclass
import random
from typing import Generator, Callable, Any

import torch
from PIL import Image, ImageDraw, ImageFont
from colorama import Fore, Style
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from torch import FloatTensor
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from compel import Compel


def clean_filename(filename):
    """
    removes all non-alphanumeric characters from a string so it is safe to use as a filename
    """
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

@dataclass
class SampleRequest:
    prompt: str
    negative_prompt: str
    seed: int
    size: tuple[int,int]
    wants_random_caption: bool = False

    def __str__(self):
        rep = self.prompt
        if len(self.negative_prompt) > 0:
            rep += f"\n negative prompt: {self.negative_prompt}"
        rep += f"\n seed: {self.seed}"
        return rep


def chunk_list(l: list, batch_size: int,
               compatibility_test: Callable[[Any,Any], bool]=lambda x,y: True
               ) -> Generator[list, None, None]:
    buckets = []
    for item in l:
        compatible_bucket = next((b for b in buckets if compatibility_test(item, b[0])), None)
        if compatible_bucket is not None:
            compatible_bucket.append(item)
        else:
            buckets.append([item])

    for b in buckets:
        for i in range(0, len(b), batch_size):
            yield b[i:i + batch_size]


def get_best_size_for_aspect_ratio(aspect_ratio, default_resolution) -> tuple[int, int]:
    sizes = []
    target_pixel_count = default_resolution * default_resolution
    for w in range(256, 1024, 64):
        for h in range(256, 1024, 64):
            if abs((w * h) - target_pixel_count) <= 128 * 64:
                sizes.append((w, h))
    best_size = min(sizes, key=lambda s: abs(1 - (aspect_ratio / (s[0] / s[1]))))
    return best_size


class SampleGenerator:
    seed: int
    default_resolution: int
    cfgs: list[float] = [7, 4, 1.01]
    scheduler: str = 'ddim'
    num_inference_steps: int = 30
    random_captions = False

    sample_requests: [str]
    log_folder: str
    log_writer: SummaryWriter

    def __init__(self,
                 log_folder: str,
                 log_writer: SummaryWriter,
                 default_resolution: int,
                 config_file_path: str,
                 batch_size: int,
                 default_seed: int,
                 default_sample_steps: int,
                 use_xformers: bool,
                 use_penultimate_clip_layer: bool):
        self.log_folder = log_folder
        self.log_writer = log_writer
        self.batch_size = batch_size
        self.config_file_path = config_file_path
        self.use_xformers = use_xformers
        self.show_progress_bars = False
        self.generate_pretrain_samples = False
        self.use_penultimate_clip_layer = use_penultimate_clip_layer

        self.default_resolution = default_resolution
        self.default_seed = default_seed
        self.sample_steps = default_sample_steps

        self.sample_requests = None
        self.reload_config()
        print(f" * SampleGenerator initialized with {len(self.sample_requests)} prompts, generating samples every {self.sample_steps} training steps, using scheduler '{self.scheduler}' with {self.num_inference_steps} inference steps")
        if not os.path.exists(f"{log_folder}/samples/"):
            os.makedirs(f"{log_folder}/samples/")

    def reload_config(self):
        try:
            config_file_extension = os.path.splitext(self.config_file_path)[1].lower()
            if config_file_extension == '.txt':
                self._reload_sample_prompts_txt(self.config_file_path)
            elif config_file_extension == '.json':
                self._reload_config_json(self.config_file_path)
            else:
                raise ValueError(f"Unrecognized file type '{config_file_extension}' for sample config, must be .txt or .json")
        except Exception as e:
            logging.warning(
                f" * {Fore.LIGHTYELLOW_EX}Error trying to read sample config from {self.config_file_path}: {Style.RESET_ALL}{e}")
            logging.warning(
                f"    Edit {self.config_file_path} to fix the problem. It will be automatically reloaded next time samples are due to be generated."
            )
            if self.sample_requests == None:
                logging.warning(
                    f"    Will generate samples from random training image captions until the problem is fixed.")
                self.sample_requests = self._make_random_caption_sample_requests()

    def update_random_captions(self, possible_captions: list[str]):
        random_prompt_sample_requests = [r for r in self.sample_requests if r.wants_random_caption]
        for i, r in enumerate(random_prompt_sample_requests):
            r.prompt = possible_captions[i % len(possible_captions)]

    def _reload_sample_prompts_txt(self, path):
        with open(path, 'rt') as f:
            self.sample_requests = [SampleRequest(prompt=line.strip(),
                                                  negative_prompt='',
                                                  seed=self.default_seed,
                                                  size=(self.default_resolution, self.default_resolution)
                                                  ) for line in f]
            if len(self.sample_requests) == 0:
                self.sample_requests = self._make_random_caption_sample_requests()

    def _make_random_caption_sample_requests(self):
        num_random_captions = min(4, self.batch_size)
        return [SampleRequest(prompt='',
                              negative_prompt='',
                              seed=self.default_seed,
                              size=(self.default_resolution, self.default_resolution),
                              wants_random_caption=True)
                for _ in range(num_random_captions)]

    def _reload_config_json(self, path):
        with open(path, 'rt') as f:
            config = json.load(f)
            # if keys are missing, keep current values
            self.default_resolution = config.get('resolution', self.default_resolution)
            self.cfgs = config.get('cfgs', self.cfgs)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.scheduler = config.get('scheduler', self.scheduler)
            self.num_inference_steps = config.get('num_inference_steps', self.num_inference_steps)
            self.show_progress_bars = config.get('show_progress_bars', self.show_progress_bars)
            self.generate_pretrain_samples = config.get('generate_pretrain_samples', self.generate_pretrain_samples)
            self.sample_steps = config.get('generate_samples_every_n_steps', self.sample_steps)
            sample_requests_config = config.get('samples', None)
            if sample_requests_config is None:
                self.sample_requests = self._make_random_caption_sample_requests()
            else:
                default_seed = config.get('seed', self.default_seed)
                self.sample_requests = [SampleRequest(prompt=p.get('prompt', ''),
                                                      negative_prompt=p.get('negative_prompt', ''),
                                                      seed=p.get('seed', default_seed),
                                                      size=tuple(p.get('size', None) or
                                                                 get_best_size_for_aspect_ratio(p.get('aspect_ratio', 1), self.default_resolution)),
                                                      wants_random_caption=p.get('random_caption', False)
                                                      ) for p in sample_requests_config]
            if len(self.sample_requests) == 0:
                self.sample_requests = self._make_random_caption_sample_requests()

    @torch.no_grad()
    def generate_samples(self, pipe: StableDiffusionPipeline, global_step: int):
        """
        generates samples at different cfg scales and saves them to disk
        """
        disable_progress_bars = not self.show_progress_bars

        try:
            font = ImageFont.truetype(font="arial.ttf", size=20)
        except:
            font = ImageFont.load_default()

        if not self.show_progress_bars:
            print(f" * Generating samples at gs:{global_step} for {len(self.sample_requests)} prompts")

        sample_index = 0
        with autocast():
            batch: list[SampleRequest]
            def sample_compatibility_test(a: SampleRequest, b: SampleRequest) -> bool:
                return a.size == b.size
            batches = list(chunk_list(self.sample_requests, self.batch_size,
                                    compatibility_test=sample_compatibility_test))
            pbar = tqdm(total=len(batches), disable=disable_progress_bars, position=1, leave=False,
                              desc=f"{Fore.YELLOW}Image samples (batches of {self.batch_size}){Style.RESET_ALL}")
            compel = Compel(tokenizer=pipe.tokenizer,
                            text_encoder=pipe.text_encoder,
                            use_penultimate_clip_layer=self.use_penultimate_clip_layer)
            for batch in batches:
                prompts = [p.prompt for p in batch]
                negative_prompts = [p.negative_prompt for p in batch]
                seeds = [(p.seed if p.seed != -1 else random.randint(0, 2 ** 30))
                         for p in batch]
                # all sizes in a batch are the same
                size = batch[0].size
                generators = [torch.Generator(pipe.device).manual_seed(seed) for seed in seeds]

                batch_images = []
                for cfg in self.cfgs:
                    pipe.set_progress_bar_config(disable=disable_progress_bars, position=2, leave=False,
                                                 desc=f"{Fore.LIGHTYELLOW_EX}CFG scale {cfg}{Style.RESET_ALL}")
                    prompt_embeds = compel(prompts)
                    negative_prompt_embeds = compel(negative_prompts)
                    images = pipe(prompt_embeds=prompt_embeds,
                                  negative_prompt_embeds=negative_prompt_embeds,
                                  num_inference_steps=self.num_inference_steps,
                                  num_images_per_prompt=1,
                                  guidance_scale=cfg,
                                  generator=generators,
                                  width=size[0],
                                  height=size[1],
                                  ).images

                    for image in images:
                        draw = ImageDraw.Draw(image)
                        print_msg = f"cfg:{cfg:.1f}"

                        l, t, r, b = draw.textbbox(xy=(0, 0), text=print_msg, font=font)
                        text_width = r - l
                        text_height = b - t

                        x = float(image.width - text_width - 10)
                        y = float(image.height - text_height - 10)

                        draw.rectangle((x, y, image.width, image.height), fill="white")
                        draw.text((x, y), print_msg, fill="black", font=font)

                    batch_images.append(images)
                    del images

                del generators
                #print("batch_images:", batch_images)

                width = size[0] * len(self.cfgs)
                height = size[1]

                for prompt_idx in range(len(batch)):
                    #print(f"batch_images[:][{prompt_idx}]: {batch_images[:][prompt_idx]}")
                    result = Image.new('RGB', (width, height))
                    x_offset = 0

                    for cfg_idx in range(len(self.cfgs)):
                        image = batch_images[cfg_idx][prompt_idx]
                        result.paste(image, (x_offset, 0))
                        x_offset += image.width

                    prompt = prompts[prompt_idx]
                    clean_prompt = clean_filename(prompt)

                    result.save(f"{self.log_folder}/samples/gs{global_step:05}-{sample_index}-{clean_prompt[:100]}.jpg", format="JPEG", quality=95, optimize=True, progressive=False)
                    with open(f"{self.log_folder}/samples/gs{global_step:05}-{sample_index}-{clean_prompt[:100]}.txt", "w", encoding='utf-8') as f:
                        f.write(str(batch[prompt_idx]))

                    tfimage = transforms.ToTensor()(result)
                    if batch[prompt_idx].wants_random_caption:
                        self.log_writer.add_image(tag=f"sample_{sample_index}", img_tensor=tfimage, global_step=global_step)
                    else:
                        self.log_writer.add_image(tag=f"sample_{sample_index}_{clean_prompt[:100]}", img_tensor=tfimage, global_step=global_step)
                    sample_index += 1

                    del result
                    del tfimage
                del batch_images

                pbar.update(1)

    @torch.no_grad()
    def create_inference_pipe(self, unet, text_encoder, tokenizer, vae, diffusers_scheduler_config: dict):
        """
        creates a pipeline for SD inference
        """
        scheduler = self._create_scheduler(diffusers_scheduler_config)
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None, # save vram
            requires_safety_checker=None, # avoid nag
            feature_extractor=None, # must be None if no safety checker
        )
        if self.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        return pipe


    @torch.no_grad()
    def _create_scheduler(self, scheduler_config: dict):
        scheduler = self.scheduler
        if scheduler not in ['ddim', 'dpm++', 'pndm', 'ddpm', 'lms', 'euler', 'euler_a', 'kdpm2']:
            print(f"unsupported scheduler '{self.scheduler}', falling back to ddim")
            scheduler = 'ddim'

        if scheduler == 'ddim':
            return DDIMScheduler.from_config(scheduler_config)
        elif scheduler == 'dpm++':
            return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++")
        elif scheduler == 'pndm':
            return PNDMScheduler.from_config(scheduler_config)
        elif scheduler == 'ddpm':
            return DDPMScheduler.from_config(scheduler_config)
        elif scheduler == 'lms':
            return LMSDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == 'euler':
            return EulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == 'euler_a':
            return EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == 'kdpm2':
            return KDPM2AncestralDiscreteScheduler.from_config(scheduler_config)
        else:
            raise ValueError(f"unknown scheduler '{scheduler}'")
