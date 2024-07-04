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
from typing import TYPE_CHECKING, Generator, Optional, List, Tuple, Dict, Any

import torch
from torchvision import transforms

from PIL import Image
import PIL.ImageOps as ImageOps
from pynvml import *

from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig, LlavaForConditionalGeneration, \
    AutoProcessor, LlavaProcessor, AutoTokenizer, AutoModelForVision2Seq, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPast
from colorama import Fore, Style
from unidecode import unidecode

from plugins.caption_plugins import load_prompt_alteration_plugin
from utils.patch_cog import patch_cog
from utils.ed_logging import configure_logging
from data.generators import image_path_generator, SUPPORTED_EXT

try:
    from moai.load_moai import prepare_moai
except ImportError:
    print("moai not found, skipping")

Image.MAX_IMAGE_PIXELS = 715827880*4 # expand the size limit

IMAGE_SIZE_COG1: int = 490
IMAGE_SIZE_COG2: int = 1344
PATCH_SIZE: int = 14

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

def create_bnb_config(bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type= "fp4"):
    return BitsAndBytesConfig(
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=None,
        llm_int8_threshold= 6.0,
        load_in_4bit=True,
        load_in_8bit=False,
    )

class BaseModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        logging.info(f"Loading {model_name}")

    def load_model(self, dtype: str="auto"):
        bnb_config = self._maybe_create_bnb_config(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config = bnb_config
        ).to(0)

        self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
        return self.model, self.tokenizer
    
    def _maybe_create_bnb_config(self, dtype, auto_bnb=True, auto_bnb_dtype="fp4"):
        bnb_config = None
        if dtype == "auto":
            if auto_bnb:
                bnb_config = create_bnb_config(bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type=auto_bnb_dtype)
        if dtype in ["nf4", "fp4"]:
            bnb_config = create_bnb_config(bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type=dtype)
        return bnb_config

    def get_gen_kwargs(self, args):
        gen_kwargs = {
            "max_length": args.max_length,
            "do_sample": args.top_k is not None or args.top_p is not None or args.temp is not None or False,
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

        #logging.debug(gen_kwargs)

        if args.max_new_tokens is not None:
            logging.debug(f"** max_new_tokens set to {args.max_new_tokens}, ignoring max_length")
            del gen_kwargs["max_length"]

        if not gen_kwargs["do_sample"]:
            logging.debug(f"** Using greedy sampling")
            del gen_kwargs["top_k"]
            del gen_kwargs["top_p"]
            del gen_kwargs["temperature"]
        else:
            logging.debug(f"** Sampling enabled")
        return gen_kwargs
    
    def _clean_caption(self, caption, args):
        """
        Removes some nonsense Llava adds.
        """
        if not args.no_clean:
            logging.debug(f"**Llava pre-cleaning caption: {caption}")
            caption = caption.replace("**", "")
            caption = re.sub(r"The image does not contain .*?\.", "", caption)
            caption = re.sub(r"Please note that this description is based on .*?\.", "", caption)
            caption = re.sub(r", adding to .*? overall appearance", "", caption)
            caption = re.sub(r"The rest of .*? is not visible in the image, focusing .*?\.", "", caption)
            caption = re.sub(r", adding to the .*? of the image", "", caption)
            caption = re.sub(r", making .*? the focal point of the image", "", caption)
            caption = re.sub(r", adding .*? to the scene", "", caption)
            caption = re.sub(r", adding an element of .*? to .*?\.",".", caption) # [intrigue, color, etc] .. [the image, the scene, etc]
            caption = re.sub(r", hinting at .*?\.", ".", caption)
            caption = re.sub(r"hinting at .*?\.", ".", caption)
            caption = re.sub(r", .*? is the main subject of the .*?\.",".", caption) # [who, which, etc] .. [image, photo, etc]
            caption = re.sub(r", .*? is the main subject of the .*?,",".", caption)
            caption = caption.replace(", who is the main subject of the image,", "")
            caption = caption.replace(", which is the main subject of the image,", "")
            caption = caption.replace(", who is the main subject of the photo.", ".")
            caption = caption.replace(", who is the main subject.", ".")
            caption = caption.replace("who is the main subject.", ".")
            caption = caption.replace(", who is the central focus of the composition.", ".")
            caption = caption.replace("who is the central focus of the composition.", ".")
            caption = self._truncate_to_whole_sentences(caption)

            logging.debug(f"**Llava post-cleaning caption: {caption}")
        return caption

    def caption(prompt, args):
        return ""

class XtunerLlavaModelManager(BaseModelWrapper):
    def __init__(self, model_name: str="xtuner/llava-llama-3-8b-v1_1-transformers"):
        self.model_name = "xtuner/llava-llama-3-8b-v1_1-transformers"
        super().__init__(model_name)
        logging.info("Loading Xtuner Llava-Llama3 model...")

    def load_model(self, dtype="auto"):
        bnb_config = self._maybe_create_bnb_config(dtype, auto_bnb=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config
        ).to("cuda")

        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers")

        return self.model, self.tokenizer

    def get_inputs(self, image: Image.Image, prompt: str):
        inputs = self.processor(prompt, image, return_tensors='pt').to(0, torch.float16)
        return inputs

    def _build_conversational_input_ids(self, prompt, starts_with):
        return (f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|>"
          f"<|start_header_id|>assistant<|end_header_id|>\n\n{starts_with}")

    def _truncate_to_whole_sentences(self, caption):
        # model does not stop generating cleanly and cuts off mid sentence
        caption = caption.split(".")
        caption = ". ".join(caption[0:-1]) + "."
        caption = caption.replace("\n","")
        caption = caption.replace("  "," ")
        return caption

    def caption(self, prompt, image, args, force_words_ids, bad_words_ids, history=[]):
        gen_kwargs = self.get_gen_kwargs(args)

        prompt = self._build_conversational_input_ids(prompt, args.starts_with)
        inputs = self.processor(prompt, image, return_tensors='pt').to(0, torch.float16)
        # inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        # inputs['input_ids'].shape: torch.Size([1, 34])
        # inputs['attention_mask'].shape: torch.Size([1, 34])
        # inputs['pixel_values'].shape: torch.Size([1, 3, 336, 336])
        
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs['attention_mask'],
            "pixel_values": inputs['pixel_values'],
            #"images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)] for _ in range(args.num_beams)],
            #"output_hidden_states": True,
            #"return_dict": True
        }
        len_inputs = inputs['input_ids'].shape[1]

        outputs = self.model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)

        caption = self.processor.decode(outputs[0][len_inputs:], skip_special_tokens=True)

        caption = self._clean_caption(caption, args)
        return caption

# class MoaiManager:
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#         self.moai_model = None
#         self.moai_processor = None
#         self.seg_model = None
#         self.seg_processor = None
#         self.od_model = None
#         self.od_processor = None
#         self.sgg_model = None
#         self.ocr_model = None
#         logging.info("Loading Moai model...")

#     def load_model(self, bits: int=4, grad_ckpt: bool=False, lora: bool=False, dtype: str="fp16"):
#         moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model \
#             = prepare_moai(moai_path=self.model_name, bits=bits, grad_ckpt=grad_ckpt, lora=lora, dtype=dtype)
#         self.moai_model = moai_model
#         self.moai_processor = moai_processor
#         self.seg_model = seg_model
#         self.seg_processor = seg_processor
#         self.od_model = od_model
#         self.od_processor = od_processor
#         self.sgg_model = sgg_model
#         self.ocr_model = ocr_model

#         return moai_model, moai_processor

#     def get_inputs(self, image: Image.Image, prompt: str):
#         moai_inputs = self.moai_model.demo_process(image=image,
#                                     prompt=prompt,
#                                     processor=self.moai_processor,
#                                     seg_model=self.seg_model,
#                                     seg_processor=self.seg_processor,
#                                     od_model=self.od_model,
#                                     od_processor=self.od_processor,
#                                     sgg_model=self.sgg_model,
#                                     ocr_model=self.ocr_model,
#                                     device="cuda:0")
#         return moai_inputs

class CogGLMManager(BaseModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if not model_name:
            self.model_name = "THUDM/cogglm-6b"
        else:
            self.model_name = model_name
        logging.info("Loading CogGLM model...")

    def load_model(self, dtype: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bnb_config = None
        if dtype in ["auto","nf4"]:
            bnb_config = create_bnb_config()
        self.model = model = AutoModelForCausalLM.from_pretrained(
            "THUDM/glm-4v-9b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=bnb_config
        ).eval()
        if bnb_config is None:
            # if BNB is used it is automatically sent to cuda device, otherwise need to move it manually
            self.model = model.to("cuda")

    def caption(self, prompt, image, args, force_words_ids, bad_words_ids, history=[]):
        gen_kwargs = self.get_gen_kwargs(args)

        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
        inputs.to("cuda")

        outputs = self.model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)

        len_inputs = inputs['input_ids'].shape[1]
        outputs_without_prompt = outputs[:, len_inputs:]

        caption = self.tokenizer.decode(outputs_without_prompt[0], skip_special_tokens=True)
        return caption

class LlavaNextManager(BaseModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def load_model(self, dtype: str = "auto"):
        self.tokenizer = LlamaTokenizer.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model.to("cuda")

    def caption(self, prompt, image, args, force_words_ids, bad_words_ids, history=[]):
        gen_kwargs = self.get_gen_kwargs(args)
        image_marker = "<image>"
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {image_marker}\n{prompt} ASSISTANT:"
        prompt_len = len(prompt) - len(image_marker)
        prompt = prompt + args.starts_with
        print(f"prompt: {prompt}")
        print(f"image: {image}")
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")

        output = self.model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        print(f"raw return: {caption}")
        caption = caption[prompt_len:]
        if args.remove_starts_with:
            caption = caption[len(args.starts_with):].strip()
        return caption

# class AutoProcessAndModelManager(BaseModelWrapper):
#     def __init__(self, model_name: str):
#         super().__init__(model_name)

#     def load_model(self, dtype: str = "auto"):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.processor = AutoProcessor.from_pretrained(self.model_name)
#         # bnb_config = None
#         # bnb_config = self._maybe_create_bnb_config(dtype, auto_bnb_dtype="nf4")
#         #print(bnb_config)
#         self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, quantization_config=create_bnb_config()).eval()
#         # if bnb_config is None:
#         #     self.model.to("cuda", dtype=torch.float16)

#     def caption(self, prompt, image, args, force_words_ids, bad_words_ids, history=[]):
#         messages = [{"role": "user","content": [{"type": "image"},{"type": "text", "text": prompt},]}]
#         prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
#         inputs = {k: v.to("cuda") for k, v in inputs.items()}

#         gen_kwargs = self.get_gen_kwargs(args)

#         generated_ids = self.model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)
#         generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
#         print(type(generated_texts))
#         print(len(generated_texts))
#         print(generated_texts[0])
#         caption = generated_texts[0].split("Assistant:")[-1]

#         return caption


class CogVLMManager(BaseModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if not model_name:
            self.model_name = "THUDM/cogvlm-chat-hf"
            self.cog_version = 1
        elif model_name.lower() == "THUDM/cogvlm2-llama3-chat-19b".lower():
            self.model_name = "THUDM/cogvlm2-llama3-chat-19b"
            self.cog_version = 2
        else:
            self.model_name = model_name
            self.cog_version = 1
        patch_cog() # fixes inv_freq key error with cogvlm, quantization, and newer transformers revisions
        logging.info("Loading CogVLM model...")

    def load_model(self, dtype: str = "auto"):
        if self.model_name.lower() == "THUDM/cogvlm-chat-hf".lower():
            self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        elif self.model_name.lower() == "THUDM/cogvlm2-llama3-chat-19b".lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token_id = 128002 # for Llama 3
        else:
            raise ValueError("Unknown model name")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=create_bnb_config()
        )
        return self.model, self.tokenizer

    def _build_conversation_input_ids(self,
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List[Image.Image]] = None,
            starts_with: Optional[str] = None,
    ):
        # based on https://huggingface.co/THUDM/cogvlm-chat-hf/blob/main/modeling_cogvlm.py
        image_size: int = IMAGE_SIZE_COG2 if self.cog_version == 2 else IMAGE_SIZE_COG1
        patch_size: int = PATCH_SIZE
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []

        text = f"Question: {query} Answer: "
        text += starts_with if starts_with is not None else ""

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [0] # LANGUAGE_TOKEN_TYPE
        if images is not None and len(images) == 1:
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
            if self.cog_version == 1:
                vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            elif self.cog_version == 2:
                vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2

            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [1] * vision_token_num
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [0] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "images": images,
        }

    def caption(self, prompt, image, args, force_words_ids, bad_words_ids, history=[]):
        gen_kwargs = self.get_gen_kwargs(args)

        inputs = self._build_conversation_input_ids(query=prompt, history=history, images=[image], starts_with=args.starts_with)

        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs['token_type_ids'].unsqueeze(0).to("cuda"),
            "attention_mask": inputs['attention_mask'].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)] for _ in range(args.num_beams)],
            "output_hidden_states": True,
            "return_dict": True
        }

        outputs = self.model.generate(**inputs, **gen_kwargs, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)

        len_inputs = inputs['input_ids'].shape[1]
        outputs_without_prompt = outputs[:, len_inputs:]

        caption = self.tokenizer.decode(outputs_without_prompt[0], skip_special_tokens=True)
        return caption

def get_model_wrapper(model_name: str):
    if model_name is None:
        logging.info("No model given, defaulting to CogVLM")
        return CogVLMManager("thudm/cogvlm-chat-hf")
    match model_name.casefold():
        # case x if "moai" in x:
        #     #return MoaiManager(model_name)
        #     return None
        case "xtuner/llava-llama-3-8b-v1_1-transformers":
            return XtunerLlavaModelManager(model_name)
        case "thudm/glm-4v-9b":
            return CogGLMManager(model_name)
        case "thudm/cogvlm2-llama3-chat-19b":
            return CogVLMManager(model_name)
        case x if x in ["thudm/cogvlm-chat-hf","thudm/cogagent-chat-hf"]:
            return CogVLMManager(model_name)
        case "llava-hf/llava-v1.6-vicuna-7b-hf":
            return LlavaNextManager(model_name)
        case None:
            return CogVLMManager(model_name)
        case _:
            raise ValueError(f"Model {model_name} not supported")

def get_inputs_dict(inputs):
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs['token_type_ids'].unsqueeze(0).to("cuda"),
            "attention_mask": inputs['attention_mask'].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)] for _ in range(args.num_beams)],
            "output_hidden_states": True,
            "return_dict": True
        }

def main(args):
    prompt_plugin_fn = load_prompt_alteration_plugin(args.prompt_plugin, args=args)
    model_wrapper = get_model_wrapper(args.model)
    model_wrapper.load_model()

    args.append = args.append or ""
    if len(args.append) > 0:
        args.append = " " + args.append.strip()

    gen_kwargs = model_wrapper.get_gen_kwargs(args)

    force_words_ids = None
    if args.force_words is not None:
        force_words = args.force_words.split(",") if args.force_words is not None else []
        logging.info(f"** force_words: {Fore.LIGHTGREEN_EX}{force_words}{Style.RESET_ALL}")
        # if args.model contains "cog"
        if "cog" in args.model:
            force_words_ids = model_wrapper.tokenizer(force_words, add_special_tokens=False)["input_ids"] if force_words else []
        else:
            force_words_ids = model_wrapper.tokenizer(force_words)["input_ids"] if force_words else []

    bad_words_ids = None
    if args.bad_words is not None:
        bad_words = args.bad_words.split(",") if args.bad_words is not None else []
        logging.info(f"** bad_words: {Fore.LIGHTGREEN_EX}{bad_words}{Style.RESET_ALL}")
        bad_words_ids = model_wrapper.tokenizer(bad_words, add_special_tokens=False)["input_ids"] if bad_words else []

    logging.info(f"** gen_kwargs: \n{Fore.LIGHTGREEN_EX}{gen_kwargs}{Style.RESET_ALL}")

    save_params(args, gen_kwargs)

    total_start_time = time.time()
    i_processed = 0

    starts_with = args.starts_with.strip() if args.starts_with is not None else ""

    for i, image_path in enumerate(image_path_generator(args.image_dir, do_recurse=not args.no_recurse)):
        candidate_caption_path = image_path.replace(os.path.splitext(image_path)[-1], ".txt")

        if args.no_overwrite and os.path.exists(candidate_caption_path):
            logging.warning(f"Skipping {image_path}, caption already exists.")
            continue

        cap_start_time = time.time()
        image = Image.open(image_path)

        try:
            image = image.convert("RGB")
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            logging.warning(f"Non-fatal error processing {image_path}: {e}")
            continue

        pixel_count = image.height * image.width
        if pixel_count < args.min_pixels:
            logging.warning(f" * Image under {args.min_pixels} pixels, skipping. Path: {image_path}")
            continue

        logging.debug(f" __ Prompt before plugin: {Fore.LIGHTGREEN_EX}{args.prompt}{Style.RESET_ALL}")
        prompt = prompt_plugin_fn(image_path, args=args)
        logging.debug(f" __ Modified prompt after plugin: {Fore.LIGHTGREEN_EX}{prompt}{Style.RESET_ALL}")

        with torch.no_grad():
            #def caption(self, prompt, images, args, force_words_ids, bad_words_ids, history=[]):
            caption = model_wrapper.caption(prompt, image, args, force_words_ids=force_words_ids, bad_words_ids=bad_words_ids)

            if not args.remove_starts_with:
                # deal with caption starting with comma, etc
                if not re.match(r"^\W", caption):
                    caption = starts_with + " " + caption
                else:
                    caption = starts_with + caption

            caption += args.append
            
            if not args.no_clean:
                caption = unidecode(caption)

            with open(candidate_caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
            vram_gb = get_gpu_memory_map()
            elapsed_time = time.time() - cap_start_time
            logging.info(f"n:{i:05}, VRAM: {Fore.LIGHTYELLOW_EX}{vram_gb:0.1f} GB{Style.RESET_ALL}, elapsed: {Fore.LIGHTYELLOW_EX}{elapsed_time:0.1f}{Style.RESET_ALL} sec, sqrt_pixels: {pow(float(pixel_count),0.5):0.1f}, Captioned {Fore.LIGHTYELLOW_EX}{image_path}{Style.RESET_ALL}: ")
            logging.info(f"{Fore.LIGHTCYAN_EX}{caption}{Style.RESET_ALL}")
            i_processed += 1

    if i_processed == 0:
        logging.info(f"** No images found in {args.image_dir} with extension in {SUPPORTED_EXT} OR no images left to caption (did you use --no_overwrite?)")
        exit(1)

    total_elapsed_time = time.time() - total_start_time
    avg_time = total_elapsed_time / i_processed
    hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))
    logging.info(f"** Done captioning {args.image_dir} with prompt '{prompt}', total elapsed: {hh_mm_ss} (hh_mm_ss), avg: {avg_time:0.1f} sec/image")


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
    Using Max may abruptly end caption, consider modifying prompt or use length_penalty instead. Some models react differently to these settings.

Find more info on the Huggingface Transformers documentation: https://huggingface.co/docs/transformers/main_classes/text_generation
Parameters definitions and use map directly to their API.
"""

DESCRIPTION = f"** {Fore.LIGHTBLUE_EX}CogVLM captioning script{Style.RESET_ALL} **\n Use --help for usage."

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size for batch processing. Does NOT work with COG! (def: 1)")
    argparser.add_argument("--debug", action="store_true", help="Enable debug logging")
    argparser.add_argument("--disable_4bit", action="store_true", help="Disables 4bit inference for compatibility or experimentation. Bad for VRAM, fallback is bf16.")
    argparser.add_argument("--dtype", choices=["auto","fp16","bf16","nf4","fp4"], default="auto", help="Data type for inference (def: auto, see docs)")
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
    argparser.add_argument("--no_clean", action="store_true", help="Skips cleaning of \"junk\" phrases")
    argparser.add_argument("--force_words", type=str, default=None, help="Forces the model to include these words in the caption, use CSV format.")
    argparser.add_argument("--bad_words", type=str, default=None, help="Words that will not be allowed, use CSV format.")
    argparser.add_argument("--append", type=str, default=None, help="Extra string to append to all captions. ex. 'painted by John Doe'")
    argparser.add_argument("--no_recurse", action="store_true", help="Do not recurse into subdirectories.")
    argparser.add_argument("--prompt_plugin", type=str, default=None, help="Function name to modify prompt, edit code to add plugins.")
    argparser.add_argument("--starts_with", type=str, default=None, help="Force start words on the output caption.")
    argparser.add_argument("--remove_starts_with", action="store_true", help="Removes the starts_with words from the output caption.")
    argparser.add_argument("--append_log", action="store_true", help="Sets logging to append mode.")
    argparser.add_argument("--model", type=str, default=None, help="Model to use for captioning.")
    argparser.add_argument("--min_pixels", type=int, default=1, help="Minimum total pixel size to caption, under the limit will be skipped")
    args, unknown_args = argparser.parse_known_args()

    configure_logging(args, "caption_cog.log")

    unknown_args_dict = {}
    print(unknown_args)
    print(len(unknown_args))
    for i in range(0, len(unknown_args)-1, 1):
        key = unknown_args[i].lstrip('-')
        if unknown_args[i+1].startswith("-"): # "store_true" instead of a kvp
            value = True
        else:
            value = unknown_args[i+1] # value is next item for all kvp in unknown args
            i += 1  # skip over the value of the kvp for next iteration to get next key
        unknown_args_dict[key] = value
        setattr(args, key, value)  # Add each unknown argument to the args namespace

    logging.info(f"** Unknown args have been added to args for plugins: {Fore.LIGHTGREEN_EX}{unknown_args_dict}{Style.RESET_ALL}")

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
