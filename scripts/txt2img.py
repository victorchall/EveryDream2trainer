"""
Copyright [2022] Victor C Hall

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

import argparse
import os
import random
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverSDEScheduler
from torch.cuda.amp import autocast
from transformers import CLIPTextModel, CLIPTokenizer

def __generate_sample(pipe: StableDiffusionPipeline, prompt: str, cfg: float, height: int, width: int, gen,
                      steps: int = 30, batch_size: int = 1):
    """
    generates a single sample at a given cfg scale and saves it to disk
    """
    with autocast():
        images = pipe(prompt,
                      num_inference_steps=steps,
                      num_images_per_prompt=batch_size,
                      guidance_scale=cfg,
                      generator=gen,
                      height=height,
                      width=width,
                      ).images

    return images


def __create_inference_pipe(unet, text_encoder, tokenizer, scheduler, vae):
    """
    creates a pipeline for SD inference
    """
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,  # save vram
        requires_safety_checker=None,  # avoid nag
        feature_extractor=None,  # must be none of no safety checker
    )

    return pipe


def main(args):
    # Create output folder if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    text_encoder = CLIPTextModel.from_pretrained(args.diffusers_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.diffusers_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.diffusers_path, subfolder="unet")
    sample_scheduler = DDIMScheduler.from_pretrained(args.diffusers_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.diffusers_path, subfolder="tokenizer", use_fast=False)

    text_encoder.eval()
    vae.eval()
    unet.eval()

    text_encoder.to("cuda")
    vae.to("cuda")
    unet.to("cuda")

    pipe = __create_inference_pipe(unet, text_encoder, tokenizer, sample_scheduler, vae)

    for _ in range(args.batch_count):
        seed = args.seed if args.seed != -1 else random.randint(0, 2 ** 30)
        gen = torch.Generator(device="cuda").manual_seed(seed)

        images = __generate_sample(pipe, args.prompt, args.cfg_scale, args.height, args.width, gen=gen,
                                   steps=args.steps,
                                   batch_size=args.batch_size)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for i, img in enumerate(images):
            img.save(
                f"output/img_{args.prompt[0:210].replace(' ', '_')}_cfg_{args.cfg_scale}_{i}_{seed}_{timestamp}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusers_path', type=str, default=None, required=True,
                        help='path to diffusers model (from logs)')
    parser.add_argument('--prompt', type=str, required=True, help='prompt to use')
    parser.add_argument('--height', type=int, default=512, help='height (def: 512)')
    parser.add_argument('--width', type=int, default=512, help='width (def: 512)')
    parser.add_argument('--seed', type=int, default=-1, help='seed, use -1 for random (def: -1)')
    parser.add_argument('--steps', type=int, default=50, help='inference, denoising steps (def: 50)')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='unconditional guidance scale (def: 7.5)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (def: 1)')
    parser.add_argument('--batch_count', type=int, default=1, help='batch count (def: 1)')
    args = parser.parse_args()

    main(args)
