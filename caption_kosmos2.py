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

from PIL import Image
from pynvml import *
from transformers import AutoProcessor, AutoModelForVision2Seq


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
    else:
        return b

def main(args):
     for root, dirs, files in os.walk(args.data_root):
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                start_time = time.time()

                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)
                model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
                processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)

                inputs = processor(text=GROUNDING+args.prompt, images=image, return_tensors="pt")

                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )

                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]        
                processed_text, entities = processor.post_process_generation(generated_text) # remove remaining special tokens to get just the caption and entities 

                if not args.keep_prompt:
                    processed_text = remove_starting_string(args.prompt, processed_text)

                print(f"File: {image}, Generated caption: {processed_text}")

                name = os.path.splitext(full_file_path)[0]
                if not os.path.exists(f"{name}.txt") or args.over_write:
                    with open(f"{name}.txt", "w") as f:
                        f.write(processed_text)

                if args.save_entities and (not os.path.exists(f"{name}.ent") or args.over_write):
                    with open(f"{name}.ent", "w") as entities_file:
                        entities_file.write(entities)

if __name__ == "__main__":
    print("Kosmos-2 captioning script")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="input", help="Path to folder of images to caption")
    parser.add_argument("--prompt", type=str, default="An image of", help="Prompt for generating caption")
    parser.add_argument("--keep_prompt", action="store_true", default=False, help="will keep the prompt at the start of the caption when saved")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--save_entities", action="store_true", default=False, help="Save coord box with entities to a separate .ent file")
    parser.add_argument("--over_write", action="store_true", default=False, help="will overwrite txt and ent files if they exist")
    args = parser.parse_args()
    main(args)