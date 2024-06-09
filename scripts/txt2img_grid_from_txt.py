from diffusers import StableDiffusionPipeline
import torch
from torch.cuda.amp import autocast
import os

import argparse
from PIL import Image

def __generate_sample(pipe: StableDiffusionPipeline, prompt, cfg: float, height: int, width: int, gen,
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

def simple():
    pipe = StableDiffusionPipeline.from_pretrained("/mnt/nvme/ed2old/logs/mt_kanji-20231125-185312/ckpts/mt_kanji-ep60-gs95400").to("cuda")
    images = __generate_sample(pipe, "bicycle", cfg=7.5, height=512, width=512, gen=None, steps=40, batch_size=1)
    images[0].save("test.png")

if __name__ == "__main__":
    #simple()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_file", type=str, required=False)
    argparser.add_argument("--val_data_path", type=str, default="/mnt/nvme/mt/val", required=False)
    argparser.add_argument("--models", nargs="+", help="names of models")
    args = argparser.parse_args()
    args.val_data_path = "/mnt/nvme/mt/val/00b42"

    W = 512
    H = 512
    BATCH_SIZE = 4

    print(f"Generating grid image for {len(args.models)} models and {args.prompt_file}")
    #print each args.models
    print("Models:")
    for m in args.models:
        print(f"   {m}")

    # with open(args.prompt_file, "r") as f:
    #     prompt_master_list = []
    #     for x, line in enumerate(f):
    #         prompt_master_list.append(line.strip())
    
    # open the txt files in args.val_data_path
    prompt_master_list = {}
    for f in os.listdir(args.val_data_path):
        if f.endswith(".txt"):            
            txt_path = os.path.join(args.val_data_path, f)
            with open(os.path.join(args.val_data_path, f), "r", encoding="utf-8") as f2:
                img_path = os.path.splitext(f)[0] + ".png"
                img_path = os.path.join(args.val_data_path, img_path)
                prompt_master_list[img_path] = f2.readline().strip()
    
    print(f"Found {len(prompt_master_list)} images in {args.val_data_path}")
    print(f"First 10 images: {list(prompt_master_list.values())[:10]}")
    print()

    num_lines = len(prompt_master_list)
    grid_h = (num_lines + 1) * W  # num images plus blank for left column labels
    grid_w = (1 + len(args.models)) * H  # num models plus blank for top row labels
    grid_img = Image.new("RGB", (grid_w, grid_h))

    #num_iterations = len(prompt_master_list) // BATCH_SIZE + (len(prompt_master_list) % BATCH_SIZE > 0)

    chunked_dict_list = []
    chunk = {}
    for key, value in prompt_master_list.items():
        chunk[key] = value
        if len(chunk) == BATCH_SIZE:
            chunked_dict_list.append(chunk)
            chunk = {}

    # Append any remaining items if the total number of items is not a multiple of chunk_size
    if chunk:
        chunked_dict_list.append(chunk)

    # Iterate through the chunks
    for i, chunk in enumerate(chunked_dict_list):
        print(f"Chunk {i + 1}: {chunk}")
    exit()

    for i_m, model in enumerate(args.models):
        for j_p in range(chunked_dict_list):
            start_index = j_p * BATCH_SIZE
            end_index = (j_p + 1) * BATCH_SIZE
            current_prompts = prompt_master_list[start_index:end_index]

            print(f"{model}: {current_prompts}")
            print()

            if True:
                pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda")
                seed_generator = torch.Generator(pipe.device).manual_seed(555)
                images = __generate_sample(pipe, current_prompts, cfg=7.5, height=512, width=512, gen=seed_generator, steps=40, batch_size=BATCH_SIZE)
                # paste each image into the grid starting from H,W and incrementing by W
                for k, k_img in enumerate(images):
                    k_img.save(f"tmp/{i_m}_{k}.png")
                    grid_img.paste(k_img, (W+k*W, H+H*i_m))
                # save the grid image
                grid_img.save(f"tmp/grid.png")

            