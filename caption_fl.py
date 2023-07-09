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
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

import torch
from pynvml import *

import time
from colorama import Fore, Style


SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_gpu_memory_map():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used/1024/1024

def remove_duplicates(string):
    words = string.split(', ')  # Split the string into individual words
    unique_words = []

    for word in words:
        if word not in unique_words:
            unique_words.append(word)
        else:
            break  # Stop appending words once a duplicate is found

    return ', '.join(unique_words)

def get_examples(example_root, image_processor):
    examples = []
    for root, dirs, files in os.walk(example_root):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in SUPPORTED_EXT:
                #get .txt file of same base name
                txt_file = os.path.splitext(file)[0] + ".txt"
                with open(os.path.join(root, txt_file), 'r') as f:
                    caption = f.read()
                image = Image.open(os.path.join(root, file))
                vision_x = [image_processor(image).unsqueeze(0)]
                #vision_x = torch.cat(vision_x, dim=0)
                #vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                examples.append((caption, vision_x))
    for x in examples:
        print(f" ** Example: {x[0]}")
    return examples

def get_dtype_for_cuda_device(device):
    # check compute capability
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return dtype


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = get_dtype_for_cuda_device(device) if device == "cuda" else torch.float32

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "<image>: "
    print(f" using prompt:  {prompt}")

    if "mpt7b" in args.model:
        lang_encoder_path="anas-awadalla/mpt-7b"
        tokenizer_path="anas-awadalla/mpt-7b"
    elif "mpt1b" in args.model:
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b"
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b"

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=1,
    )

    tokenizer.padding_side = "left"

    checkpoint_path = hf_hub_download(args.model, "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(f"GPU memory used, before loading model: {get_gpu_memory_map()} MB")
    model.to(0, dtype=dtype)
    print(f"GPU memory used, after loading model: {get_gpu_memory_map()} MB")

    # examples give few shot learning for captioning the novel image
    examples = get_examples(args.example_root, image_processor)

    prompt = ""
    output_prompt = "Output:"
    per_image_prompt = "<image> " + output_prompt

    for example in iter(examples):
        prompt += f"{per_image_prompt}{example[0]}<|endofchunk|>"
    prompt += per_image_prompt # prepare for novel example
    prompt = prompt.replace("\n", "") # in case captions had newlines
    print(f" \n** Final full prompt with example pairs: {prompt}")

    # os.walk all files in args.data_root recursively
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                start_time = time.time()

                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)

                vision_x = [vx[1][0] for vx in examples]
                vision_x.append(image_processor(image).unsqueeze(0))
                vision_x = torch.cat(vision_x, dim=0)
                vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                vision_x = vision_x.to(device, dtype=dtype)

                lang_x = tokenizer(
                    [prompt], # blank for image captioning
                    return_tensors="pt",
                )
                lang_x.to(device)

                input_ids = lang_x["input_ids"].to(device)

                with torch.cuda.amp.autocast(dtype=dtype), torch.no_grad():
                    generated_text = model.generate(
                        vision_x=vision_x,
                        lang_x=input_ids,
                        attention_mask=lang_x["attention_mask"],
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        num_beams=args.num_beams,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                del vision_x
                del lang_x

                # trim and clean
                generated_text = tokenizer.decode(generated_text[0][len(input_ids[0]):], skip_special_tokens=True)
                generated_text = generated_text.split(output_prompt)[0]
                generated_text = remove_duplicates(generated_text)

                exec_time = time.time() - start_time
                print(f"* Caption:   {generated_text}")
                
                print(f"  Time for last caption: {exec_time} sec.  GPU memory used: {get_gpu_memory_map()} MB")

                name = os.path.splitext(full_file_path)[0]
                if not os.path.exists(name):
                    with open(f"{name}.txt", "w") as f:
                        f.write(generated_text)
    print("Done!")

if __name__ == "__main__":
    print(f"Available models:")
    print(f"  openflamingo/OpenFlamingo-9B-vitl-mpt7b   (default)")
    print(f"  openflamingo/OpenFlamingo-3B-vitl-mpt1b")    
    print(f"  openflamingo/OpenFlamingo-4B-vitl-rpj3b")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="input", help="Path to images")
    parser.add_argument("--example_root", type=str, default="examples", help="Path to 2-3 precaptioned images to guide generation")
    parser.add_argument("--model", type=str, default="openflamingo/OpenFlamingo-9B-vitl-mpt7b", help="Model name or path")
    parser.add_argument("--force_cpu", action="store_true", default=False, help="force using CPU even if GPU is available")
    parser.add_argument("--min_new_tokens", type=int, default=20, help="minimum number of tokens to generate")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="maximum number of tokens to generate")
    parser.add_argument("--num_beams", type=int, default=8, help="number of beams, more is more accurate but slower")
    parser.add_argument("--prompt", type=str, default="Output: ", help="prompt to use for generation, default is 'Output: '")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling, 1.0 is default")
    parser.add_argument("--top_k", type=int, default=0, help="top_k sampling, 0 is default")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p sampling, 1.0 is default")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty, 1.0 is default")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="length penalty, 1.0 is default")
    args = parser.parse_args()

    print(f"** OPEN-FLAMINGO ** Captioning files in: {args.data_root}")
    print(f"** Using model: {args.model}")
    main(args)