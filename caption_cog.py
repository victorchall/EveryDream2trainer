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
from typing import Generator

import torch

from PIL import Image
from pynvml import *

from transformers import AutoModelForCausalLM, LlamaTokenizer
from colorama import Fore, Style

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def image_generator(image_dir) -> Generator[str, None, None]:
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any([file.endswith(ext) for ext in SUPPORTED_EXT]):
                yield os.path.join(root, file)

def get_gpu_memory_map():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used/1024/1024

def main(args):
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=not args.disable_4bit,
    )

    do_sample = args.num_beams > 1

    gen_kwargs = {
        "max_length": args.max_length, 
        "do_sample": do_sample, 
        "num_beams": args.num_beams, 
        "temperature": args.temp, 
        "top_k": args.top_k, 
        "top_p": args.top_p, 
        "repetition_penalty": args.repetition_penalty, 
        "no_repeat_ngram_size": args.no_repeat_ngram_size, 
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.max_new_tokens is not None:
        print(f"** max_new_tokens set to {args.max_new_tokens}, ignoring max_length")
        del gen_kwargs["max_length"]

    if not do_sample:
        print(f"** num_beams set to 1, sampling is disabled")
        del gen_kwargs["top_k"]
        del gen_kwargs["top_p"]

    force_words_ids = None
    if args.force_words is not None:
        force_words = args.force_words.split(",") if args.force_words is not None else []
        print(f"** force_words: {Fore.LIGHTGREEN_EX}{force_words}{Style.RESET_ALL}")
        force_words_ids = tokenizer(force_words, add_special_tokens=False)["input_ids"] if force_words else []

    bad_words_ids = None
    if args.bad_words is not None:
        bad_words = args.bad_words.split(",") if args.bad_words is not None else []
        print(f"** bad_words: {Fore.LIGHTGREEN_EX}{bad_words}{Style.RESET_ALL}")
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False)["input_ids"] if bad_words else []

    print(f"** gen_kwargs: \n{Fore.LIGHTGREEN_EX}{gen_kwargs}{Style.RESET_ALL}")

    total_start_time = time.time()
    i_processed = 0

    for image_path in image_generator(args.image_dir):
        candidate_caption_path = image_path.replace(os.path.splitext(image_path)[-1], ".txt")

        if args.no_overwrite and os.path.exists(candidate_caption_path):
            print(f"Skipping {image_path}, caption already exists.")
            continue

        start_time = time.time()
        image = Image.open(image_path)
        inputs = model.build_conversation_input_ids(tokenizer, query=args.prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)] for _ in range(args.num_beams)],
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)
            outputs_without_prompt = outputs[:, inputs['input_ids'].shape[1]:]
            caption = tokenizer.decode(outputs_without_prompt[0], skip_special_tokens=True)

            with open(candidate_caption_path, "w") as f:
                f.write(caption)
            vram_gb = get_gpu_memory_map()
            elapsed_time = time.time() - start_time
            print(f"VRAM: {Fore.LIGHTYELLOW_EX}{vram_gb:0.1f} GB{Style.RESET_ALL}, elapsed: {Fore.LIGHTYELLOW_EX}{elapsed_time:0.1f}{Style.RESET_ALL} sec, Captioned {Fore.LIGHTYELLOW_EX}{image_path}{Style.RESET_ALL}: ")
            print(f"{Fore.LIGHTCYAN_EX}{caption}{Style.RESET_ALL}")
            i_processed += 1

    if i_processed == 0:
        print(f"** No images found in {args.image_dir} with extension in {SUPPORTED_EXT} OR no images left to caption (did you use --no_overwrite?)")
        exit(1)

    total_elapsed_time = time.time() - total_start_time
    avg_time = total_elapsed_time / i_processed
    hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))
    print(f"** Done captioning {args.image_dir} with prompt '{args.prompt}', total elapsed: {hh_mm_ss} (hh_mm_ss), avg: {avg_time:0.1f} sec/image")

EXAMPLES = """ex.
Basic example:
  python caption_cog.py --image_dir /mnt/mydata/kyrie/ --prompt 'Describe this image in detail, including the subject matter and medium of the artwork.'

Use beam search and probabilistic sampling:
  python caption_cog.py --image_dir \"c:/users/chadley/my documents/pictures\" --prompt 'Write a description.' --max_new_tokens 75 --num_beams 4 --temp 0.9 --top_k 3 --top_p 0.9 --repetition_penalty 1.0 --no_repeat_ngram_size 0 --min_new_tokens 5\n

Force "cat" and "dog" and disallow the word "depicts":
  python caption_cog.py --image_dir /mnt/lcl/nvme/mldata/test --num_beams 3 --force_words "cat,dog" --bad_words "depicts"

Notes:
  numbeams > 1 enables probabilistic sampling, which is required for the temperature, top_k, top_p parameters to function. More beams is more opinions on the next token, but slower and more VRAM intensive as it is done in batch mode.
    Increasing num_beams has a substantial impact on VRAM and speed.  Ex beams =1 ~13.3gb, beams = 4 ~ 23.7GB
    Speed is linearly proportional to num_beams, so 4 beams is 4x slower than 1 beam.
  Max_length and max_new_tokens are mutually exclusive.  If max_new_tokens is set, max_length is ignored.
"""

DESCRIPTION = f"** {Fore.LIGHTBLUE_EX}CogVLM captioning script{Style.RESET_ALL} **\n"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)
    argparser.add_argument("--disable_4bit", action="store_true", help="Disables 4bit inference for compatibility or experimentation. Bad for VRAM, fallback is bf16.")
    argparser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")
    argparser.add_argument("--num_beams", type=int, default=2, help="Number of beams for sampling, see notes.")
    argparser.add_argument("--top_k", type=int, default=0, help="Top-k, filter k highest probability tokens before sampling")
    argparser.add_argument("--top_p", type=float, default=1.0, help="Top-p, selects from top tokens with cumulative probability >= p")
    argparser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    argparser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repetition n-gram size")
    argparser.add_argument("--min_new_tokens", type=int, default=5, help="Minimum number of tokens in returned caption.")
    argparser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of tokens in returned caption.")
    argparser.add_argument("--max_length", type=int, default=2048, help="Alternate to max_new_tokens, limits context.")
    argparser.add_argument("--prompt", type=str, default="Describe this image.", help="Prompt that will guide captioning")
    argparser.add_argument("--image_dir", type=str, default=None, help="Path to folder of images to caption")
    argparser.add_argument("--no_overwrite", action="store_true", help="Skips captioning images that already have a caption file.")
    argparser.add_argument("--force_words", type=str, default=None, help="Forces the model to include these words in the caption, use CSV format.")
    argparser.add_argument("--bad_words", type=str, default=None, help="Words that will not be allowed, use CSV format.")
    args = argparser.parse_args()

    print(DESCRIPTION)
    print(EXAMPLES)
    if args.image_dir is None:
        print(f"** {Fore.RED}Error: image_dir is required.{Style.RESET_ALL}")
        exit(1)
    print(f"** Running: {args.image_dir} with prompt '{args.prompt}'")

    main(args)
