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

from PIL import Image
import argparse
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, GitProcessor, GitForCausalLM, AutoModel, AutoProcessor

import torch
from  pynvml import *

import time
from colorama import Fore, Style

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used/1024/1024

def create_blip2_processor(model_name, device, dtype=torch.float16):
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.to(device)
    model.eval()
    print(f"BLIP2 Model loaded: {model_name}")
    return processor, model

def create_git_processor(model_name, device, dtype=torch.float16):
    processor = GitProcessor.from_pretrained(model_name)
    model = GitForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"GIT Model loaded: {model_name}")
    return processor, model

def create_auto_processor(model_name, device, dtype=torch.float16):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print("Auto Model loaded")
    return processor, model

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = torch.float32 if args.force_cpu else torch.float16

    if "salesforce/blip2-" in args.model.lower():
        model_type = "blip2"
        print(f"Using BLIP2 model: {args.model}")
        processor, model = create_blip2_processor(args.model, device, dtype)
    elif "microsoft/git-" in args.model.lower():
        model_type = "git"
        print(f"Using GIT model: {args.model}")
        processor, model = create_git_processor(args.model, device, dtype)
    else:
        # try to use auto model?  doesn't work with blip/git
        model_type = "auto"
        processor, model = create_auto_processor(args.model, device, dtype)

    print(f"GPU memory used, after loading model: {get_gpu_memory_map()} MB")

    # os.walk all files in args.data_root recursively
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)
                start_time = time.time()

                if model_type == "git":
                    inputs = processor(images=image, return_tensors="pt", max_new_tokens=args.max_new_tokens).to(device)
                else:
                    inputs = processor(images=image, return_tensors="pt", max_new_tokens=args.max_new_tokens).to(device, dtype)

                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(f"file: {file}, caption: {generated_text}")
                exec_time = time.time() - start_time
                print(f"  Time for last caption: {exec_time} sec.  GPU memory used: {get_gpu_memory_map()} MB")

                # get bare name
                name = os.path.splitext(full_file_path)[0]
                #name = os.path.join(root, name)
                if not os.path.exists(name):
                    with open(f"{name}.txt", "w") as f:
                        f.write(generated_text)

if __name__ == "__main__":
    print(f"{Fore.CYAN}** Current supported models:{Style.RESET_ALL}")
    print("     microsoft/git-base-textcaps")
    print("     microsoft/git-large-textcaps")
    print("     microsoft/git-large-r-textcaps")
    print("     Salesforce/blip2-opt-2.7b - (9GB VRAM or recommend 32GB sys RAM)")
    print("     Salesforce/blip2-opt-2.7b-coco - (9GB VRAM or recommend 32GB sys RAM)")
    print("     Salesforce/blip2-opt-6.7b - (16.5GB VRAM or recommend 64GB sys RAM)")
    print("     Salesforce/blip2-opt-6.7b-coco - (16.5GB VRAM or recommend 64GB sys RAM)")
    print()
    print(f"{Fore.CYAN} * The following will likely not work on any consumer GPUs or require huge sys RAM on CPU:{Style.RESET_ALL}")
    print("     salesforce/blip2-flan-t5-xl")
    print("     salesforce/blip2-flan-t5-xl-coco")
    print("     salesforce/blip2-flan-t5-xxl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="input", help="Path to images")
    parser.add_argument("--model", type=str, default="salesforce/blip2-opt-2.7b", help="model from huggingface, ex. 'salesforce/blip2-opt-2.7b'")
    parser.add_argument("--force_cpu", action="store_true", default=False, help="force using CPU even if GPU is available, may be useful to run huge models if you have a lot of system memory")
    parser.add_argument("--max_new_tokens", type=int, default=24, help="max length for generated captions")
    args = parser.parse_args()

    print(f"** Using model: {args.model}")
    print(f"** Captioning files in: {args.data_root}")
    main(args)