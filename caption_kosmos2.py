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

import torch

from PIL import Image
from pynvml import *
from transformers import AutoProcessor, AutoModelForVision2Seq
import colorama

GROUNDING = "<grounding>"
SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_gpu_memory_map():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used/1024/1024

def remove_starting_string(a, b):
    if b.startswith(a):
        return b[len(a):]  # Remove string A from the beginning of string B
    elif b.strip().startswith(a.strip()):
        return b.strip()[len(a.strip()):]
    else:
        return b

def main(args):
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    dtype=torch.float32

    if not args.cpu:
        if args.dtype == "fp16":
            dtype=torch.float16
        elif args.dtype == "bf16":
            dtype=torch.bfloat16
        elif args.dtype == "fp32":
            dtype=torch.float32
        model = model.to(dtype=dtype).cuda()
        print(f"Using cuda, model dtype: {model.dtype}")
    else:
        print(f"Using cpu, model dtype: {model.dtype}")

    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                start_time = time.time()

                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)


                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)

                if args.phrase_mode:
                    text = GROUNDING + "".join(["<phrase>" + x.strip() + "</phrase>" for x in args.prompt.split(",")])
                else:
                    text = GROUNDING + args.prompt

                inputs = processor(text=text, images=image, return_tensors="pt")

                with torch.cuda.amp.autocast(enabled=args.dtype != "fp32", dtype=dtype):
                    generated_ids = model.generate(
                        pixel_values=inputs["pixel_values"].cuda() if not args.cpu else inputs["pixel_values"],
                        input_ids=inputs["input_ids"].cuda() if not args.cpu else inputs["input_ids"],
                        attention_mask=inputs["attention_mask"].cuda() if not args.cpu else inputs["attention_mask"],
                        image_embeds=None,
                        image_embeds_position_mask=inputs["image_embeds_position_mask"].cuda() if not args.cpu else inputs["image_embeds_position_mask"],
                        use_cache=True,
                        max_new_tokens=args.max_new_tokens,
                    )

                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]        
                processed_text, entities = processor.post_process_generation(generated_text) # remove remaining special tokens to get just the caption and entities 

                if not args.keep_prompt:
                    processed_text = remove_starting_string(args.prompt, processed_text)

                print(f"File: {full_file_path}, Generated caption: {processed_text}")

                name = os.path.splitext(full_file_path)[0]
                if (not os.path.exists(f"{name}.txt") or args.overwrite) and not args.save_entities_only:
                    with open(f"{name}.txt", "w") as f:
                        f.write(processed_text)

                if args.save_entities and (not os.path.exists(f"{name}.ent") or args.overwrite):
                    with open(f"{name}.ent", "w") as entities_file:
                        entities_file.write(str(entities))
                gpu_mb_used = get_gpu_memory_map()
                print(f"gpu usage: {gpu_mb_used:.1f} mb, time taken: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    print("Kosmos-2 captioning script")
    parser = argparse.ArgumentParser()
    parser.description = "Kosmos-2 captioning script"
    parser.add_argument("--data_root", type=str, default="input", help="Path to folder of images to caption")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail: ", help="Prompt for generating caption")
    parser.add_argument("--phrase_mode", action="store_true", default=False, help="uses 'phrase mode' grounding, interprets prompt as csv list of phrases to ground.")
    parser.add_argument("--keep_prompt", action="store_true", default=False, help="will keep the prompt at the start of the caption when saved")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--save_entities", action="store_true", default=False, help="Save coord box with entities to a separate .ent file")
    parser.add_argument("--save_entities_only", action="store_true", default=False, help="Only save coord box with entities to a separate .ent file, do not write caption .txt")
    parser.add_argument("--overwrite", action="store_true", default=False, help="will overwrite .txt and .ent files if they exist")
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu instead of cuda")
    parser.add_argument("--dtype", type=str, default="fp16", help="force a different dtype if using GPU (fp16, bf16, fp32) (default: fp16)")
    args = parser.parse_args()
    parser.print_help()

    if args.save_entities_only:
        args.save_entities = True

    if not args.prompt.startswith(" "):
        args.prompt = " " + args.prompt

    print(f"Captioning images in {args.data_root} with prompt: {args.prompt}")
    print(f"Ideas for prompts:")    
    print(f"    Describe this image in detail:  (default)")
    print(f"    An image of ")
    print(f"    A two sentence description of this image:")
    print()
    main(args)