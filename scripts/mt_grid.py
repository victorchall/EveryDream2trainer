from diffusers import StableDiffusionPipeline
import torch
from torch.cuda.amp import autocast
import os

import argparse
from PIL import Image, ImageDraw, ImageFont

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

def generate_simple(prompt, model):
    pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda")
    images = __generate_sample(pipe, prompt, cfg=7.5, height=512, width=512, gen=None, steps=40, batch_size=1)
    return images[0]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=60)
    args = argparser.parse_args()
    epochs = args.epochs

    path = "/mnt/nvme/mt/val"
    
    model = None
    if epochs == 100:
        model1 = "/mnt/nvme/ed2old/logs/mt_kanji-20231125-185312/ckpts/mt_kanji-ep100-gs159000"
        model2 = "/mnt/q/monotype/kanji_nov2023_shortcap-20231129-152030/ckpts/kanji_nov2023_shortcap-ep100-gs159000"
    elif epochs == 80:
        model1 = "/mnt/nvme/ed2old/logs/mt_kanji-20231125-185312/ckpts/mt_kanji-ep80-gs127200"
        model2 = "/mnt/q/monotype/kanji_nov2023_shortcap-20231129-152030/ckpts/kanji_nov2023_shortcap-ep80-gs127200"
    elif epochs == 60:
        model1 = "/mnt/nvme/ed2old/logs/mt_kanji-20231125-185312/ckpts/mt_kanji-ep60-gs95400"
        model2 = "/mnt/q/monotype/kanji_nov2023_shortcap-20231129-152030/ckpts/kanji_nov2023_shortcap-ep60-gs95400"
    else:
        raise ValueError("epochs must be 100, 80, or 60")
    
    pipe1 = StableDiffusionPipeline.from_pretrained(model1).to("cuda")
    pipe2 = StableDiffusionPipeline.from_pretrained(model2).to("cuda")

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt") and not file.endswith("file_list.txt"):
                txt_path = os.path.join(root, file)
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = f.readline()
                generated_image1 = __generate_sample(pipe1, prompt, cfg=7.5, height=512, width=512, gen=None, steps=40, batch_size=1)[0]
                short_prompt = prompt.split(",")[0]
                generated_image2 = __generate_sample(pipe2, short_prompt, cfg=7.5, height=512, width=512, gen=None, steps=40, batch_size=1)[0]
                print(short_prompt)
                gt_path = txt_path.replace(".txt", ".png")
                print(f"Loading gt_path {gt_path}")

                ground_truth_image = Image.open(gt_path)
                ground_truth_image = ground_truth_image.resize((512, 512))
                
                combined_image = Image.new("RGB", (1536, 576), color=(96, 96, 96))
                combined_image.paste(ground_truth_image, (0, 0))
                combined_image.paste(generated_image1, (512, 0))
                combined_image.paste(generated_image2, (1024, 0))
                
                draw = ImageDraw.Draw(combined_image)
                font = ImageFont.truetype("/mnt/nvme/mt/NotoSansCJK-Bold.ttc", 18)
                draw.text((0, 510), f"epochs={epochs}", font=font)
                draw.text((200, 510), "↑ ground truth ↑", font=font)
                draw.text((650, 510), "↑ trained&generated full caption↑", font=font)
                draw.text((1140, 510), "↑ trained&generated short caption ↑", font=font)
                font = ImageFont.truetype("/mnt/nvme/mt/NotoSansCJK-Bold.ttc", 24)
                draw.text((100, 536), prompt, font=font)
                draw.text((1240, 537), short_prompt, font=font)                

                output_path = os.path.join("/mnt/nvme/mt", str(epochs), f"{file}_compare.png")
                print(f"Saving to {output_path}")
                combined_image.save(output_path)
