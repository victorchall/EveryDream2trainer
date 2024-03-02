"""
Copyright [2022-2023] Victor C Hall

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
import io
import argparse
import time
import json
import logging
import re
from typing import TYPE_CHECKING, Generator, Optional, List, Tuple, Literal

import torch
from torchvision import transforms

from PIL import Image
import PIL.ImageOps as ImageOps
from pynvml import *

from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedTokenizer
from colorama import Fore, Style

from plugins.caption_plugins import load_prompt_alteration_plugin

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]
IMAGE_SIZE: int = 490
PATCH_SIZE: int = 14

def build_conversation_input_ids(
            tokenizer: PreTrainedTokenizer,
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List[Image.Image]] = None,
            starts_with: Optional[str] = None,
    ):
        # based on https://huggingface.co/THUDM/cogvlm-chat-hf/blob/main/modeling_cogvlm.py
        image_size: int = IMAGE_SIZE
        patch_size: int = PATCH_SIZE
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []

        text = f"Question: {query} Answer: "
        text += starts_with if starts_with is not None else ""

        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [0]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = [transform(images[0])]
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [1] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [0] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': images,
        }

def image_generator(image_dir: str, do_recurse: bool = True) -> Generator[str, None, None]:
    if do_recurse:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                    yield os.path.join(root, file)
    else:
        for file in os.listdir(image_dir):
            if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                yield os.path.join(image_dir, file)

def get_gpu_memory_map():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used/1024/1024

def save_params(args, gen_kwargs):
    save_path = os.path.join(args.image_dir, "caption_cog_params.txt")
    args_dict = {
        "args": vars(args),
        "gen_kwargs": gen_kwargs,
    }
    pretty_print = json.dumps(args_dict, indent=4)
    with open(save_path, "w") as f:
        f.write(pretty_print)


def main(args):
    prompt_plugin_fn = load_prompt_alteration_plugin(args.prompt_plugin, args=args)

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True, # gee hope they don't get hacked or have a bad internal actor
        #revision=... # no one is actually doing this
        load_in_4bit=not args.disable_4bit,
    )

    do_sample = args.top_k is not None or args.top_p is not None or args.temp is not None
    if do_sample:
        args.top_k = args.top_k or 50
        args.top_p = args.top_p or 1.0
        args.temp = args.temp or 1.0

    args.append = args.append or ""
    if len(args.append) > 0:
        args.append = " " + args.append.strip()

    gen_kwargs = {
        "max_length": args.max_length,
        "do_sample": do_sample,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
        "temperature": args.temp,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
    }

    if args.max_new_tokens is not None:
        logging.info(f"** max_new_tokens set to {args.max_new_tokens}, ignoring max_length")
        del gen_kwargs["max_length"]

    if not do_sample:
        logging.info(f"** Using greedy sampling")
        del gen_kwargs["top_k"]
        del gen_kwargs["top_p"]
        del gen_kwargs["temperature"]
    else:
        logging.info(f"** Sampling enabled")

    force_words_ids = None
    if args.force_words is not None:
        force_words = args.force_words.split(",") if args.force_words is not None else []
        logging.info(f"** force_words: {Fore.LIGHTGREEN_EX}{force_words}{Style.RESET_ALL}")
        force_words_ids = tokenizer(force_words, add_special_tokens=False)["input_ids"] if force_words else []

    bad_words_ids = None
    if args.bad_words is not None:
        bad_words = args.bad_words.split(",") if args.bad_words is not None else []
        logging.info(f"** bad_words: {Fore.LIGHTGREEN_EX}{bad_words}{Style.RESET_ALL}")
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False)["input_ids"] if bad_words else []

    logging.info(f"** gen_kwargs: \n{Fore.LIGHTGREEN_EX}{gen_kwargs}{Style.RESET_ALL}")

    save_params(args, gen_kwargs)

    total_start_time = time.time()
    i_processed = 0

    starts_with = args.starts_with.strip() if args.starts_with is not None else ""

    for i, image_path in enumerate(image_generator(args.image_dir, do_recurse=not args.no_recurse)):
        candidate_caption_path = image_path.replace(os.path.splitext(image_path)[-1], ".txt")

        if args.no_overwrite and os.path.exists(candidate_caption_path):
            logging.warning(f"Skipping {image_path}, caption already exists.")
            continue

        cap_start_time = time.time()
        image = Image.open(image_path)

        try:
            image = image.convert('RGB')
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            logging.warning(f"Non-fatal error processing {image_path}: {e}")
            continue
        
        logging.debug(f" __ Prompt before plugin: {Fore.LIGHTGREEN_EX}{args.prompt}{Style.RESET_ALL}")
        prompt = prompt_plugin_fn(image_path, args=args)
        logging.debug(f" __ Modified prompt after plugin: {Fore.LIGHTGREEN_EX}{prompt}{Style.RESET_ALL}")

        inputs = build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[image], starts_with=args.starts_with)  # chat mode

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)] for _ in range(args.num_beams)],
        }

        with torch.no_grad():
            #input_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            #logging.debug(f"inputs decoded: {input_decoded}")
            outputs = model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)

            len_inputs = inputs['input_ids'].shape[1]
            outputs_without_prompt = outputs[:, len_inputs:]

            caption = tokenizer.decode(outputs_without_prompt[0], skip_special_tokens=True)
            if not args.remove_starts_with:
                # deal with caption starting with comma, etc
                if not re.match(r"^\W", caption):
                    caption = starts_with + " " + caption
                else:
                    caption = starts_with + caption

            caption += args.append

            with open(candidate_caption_path, "w") as f:
                f.write(caption)
            vram_gb = get_gpu_memory_map()
            elapsed_time = time.time() - cap_start_time
            logging.info(f"n:{i:05}, VRAM: {Fore.LIGHTYELLOW_EX}{vram_gb:0.1f} GB{Style.RESET_ALL}, elapsed: {Fore.LIGHTYELLOW_EX}{elapsed_time:0.1f}{Style.RESET_ALL} sec, Captioned {Fore.LIGHTYELLOW_EX}{image_path}{Style.RESET_ALL}: ")
            logging.info(f"{Fore.LIGHTCYAN_EX}{caption}{Style.RESET_ALL}")
            i_processed += 1

    if i_processed == 0:
        logging.info(f"** No images found in {args.image_dir} with extension in {SUPPORTED_EXT} OR no images left to caption (did you use --no_overwrite?)")
        exit(1)

    total_elapsed_time = time.time() - total_start_time
    avg_time = total_elapsed_time / i_processed
    hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))
    logging.info(f"** Done captioning {args.image_dir} with prompt '{prompt}', total elapsed: {hh_mm_ss} (hh_mm_ss), avg: {avg_time:0.1f} sec/image")


def configure_logging(args: argparse.Namespace):
    level = logging.INFO if not args.debug else logging.DEBUG
    filemode = "a" if args.append_log else "w"
    logging.basicConfig(filename="caption_cog.log", 
                        level=level, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filemode=filemode)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

EXAMPLES = """ex.
Basic example:
  python caption_cog.py --image_dir /mnt/mydata/kyrie/ --prompt 'Describe this image in detail, including the subject matter and medium of the artwork.'

Use probabilistic sampling by using any of top_k, top_p, or temp:
  python caption_cog.py --image_dir \"c:/users/chadley/my documents/pictures\" --prompt \"What is this?\" --top_p 0.9 
  
Use beam search and probabilistic sampling:
  python caption_cog.py --image_dir \"c:/users/chadley/my documents/pictures\" --prompt \"Write a description.\" --max_new_tokens 75 --num_beams 4 --temp 0.9 --top_k 3 --top_p 0.9 --repetition_penalty 1.0 --no_repeat_ngram_size 0 --min_new_tokens 5

Force "cat" and "dog" and disallow the word "depicts":
  python caption_cog.py --image_dir /mnt/lcl/nvme/mldata/test --num_beams 3 --force_words "cat,dog" --bad_words "depicts"

Use a lot of beams and try to control the length with length_penalty:
  python caption_cog.py --image_dir /mnt/lcl/nvme/mldata/test --num_beams 8 --length_penalty 0.8 --prompt "Write a single sentence description."

Notes:
  1. Setting top_k, top_p, or temp enables probabilistic sampling (aka "do_sample"), otherwise greedy sampling is used.
    a. num_beams 1 and do_sample false uses "greedy decoding"
    b. num_beams 1 and do_sample true uses "multinomial sampling"
    c. num_beams > 1 and do_sample true uses "beam-search multinomial sampling"
    d. num_beams > 1 and do_sample false uses "beam-search decoding"
  2. Max_length and max_new_tokens are mutually exclusive.  If max_new_tokens is set, max_length is ignored.  Default is max_length 2048 if nothing set.
    Using Max may abruptly end caption, consider modifying prompt or use length_penalty instead.

Find more info on the Huggingface Transformers documentation: https://huggingface.co/docs/transformers/main_classes/text_generation
Parameters definitions and use map directly to their API.
"""

DESCRIPTION = f"** {Fore.LIGHTBLUE_EX}CogVLM captioning script{Style.RESET_ALL} **\n Use --help for usage."

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true", help="Enable debug logging")
    argparser.add_argument("--disable_4bit", action="store_true", help="Disables 4bit inference for compatibility or experimentation. Bad for VRAM, fallback is bf16.")
    argparser.add_argument("--temp", type=float, default=None, help="Temperature for sampling")
    argparser.add_argument("--num_beams", type=int, default=2, help="Number of beams for beam search, default 1 (off)")
    argparser.add_argument("--top_k", type=int, default=None, help="Top-k, filter k highest probability tokens before sampling")
    argparser.add_argument("--top_p", type=float, default=None, help="Top-p, for sampling, selects from top tokens with cumulative probability >= p")
    argparser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    argparser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repetition n-gram size")
    argparser.add_argument("--min_new_tokens", type=int, default=5, help="Minimum number of tokens in returned caption.")
    argparser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of tokens in returned caption.")
    argparser.add_argument("--max_length", type=int, default=2048, help="Alternate to max_new_tokens, limits context.")
    argparser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty, lower values encourage shorter captions.")
    argparser.add_argument("--prompt", type=str, default="Write a description.", help="Prompt that will guide captioning")
    argparser.add_argument("--image_dir", type=str, default=None, help="Path to folder of images to caption")
    argparser.add_argument("--no_overwrite", action="store_true", help="Skips captioning images that already have a caption file.")
    argparser.add_argument("--force_words", type=str, default=None, help="Forces the model to include these words in the caption, use CSV format.")
    argparser.add_argument("--bad_words", type=str, default=None, help="Words that will not be allowed, use CSV format.")
    argparser.add_argument("--append", type=str, default=None, help="Extra string to append to all captions. ex. 'painted by John Doe'")
    argparser.add_argument("--no_recurse", action="store_true", help="Do not recurse into subdirectories.")
    argparser.add_argument("--prompt_plugin", type=str, default=None, help="Function name to modify prompt, edit code to add plugins.")
    argparser.add_argument("--starts_with", type=str, default=None, help="Force start words on the output caption.")
    argparser.add_argument("--remove_starts_with", action="store_true", help="Removes the starts_with words from the output caption.")
    argparser.add_argument("--append_log", action="store_true", help="Sets logging to append mode.")
    args = argparser.parse_args()

    configure_logging(args)

    print(DESCRIPTION)
    print(EXAMPLES)

    if args.image_dir is None:
        logging.error(f"** {Fore.RED}Error: image_dir is required.{Style.RESET_ALL}")
        exit(1)

    if not os.path.exists(args.image_dir):
        logging.error(f"** {Fore.RED}Error: image_dir {args.image_dir} does not exist.{Style.RESET_ALL}")
        exit(1)

    startprint = f"** Running: {args.image_dir} with prompt '{args.prompt}"
    if args.starts_with is not None:
        startprint += f" {args.starts_with}'"
    else:
        startprint += "'"
    startprint += f" <caption>"
    if args.append is not None:
        startprint += f", and appending: {args.append}"
    logging.info(startprint)

    main(args)
