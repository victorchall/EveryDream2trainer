import argparse
import math
import os.path
import random
import shutil
from typing import Optional

from tqdm.auto import tqdm

IMAGE_EXTENSIONS =  ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']


def gather_captioned_images(root_dir: str) -> list[tuple[str,Optional[str]]]:
    for directory, _, filenames in os.walk(root_dir):
        image_filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        for image_filename in image_filenames:
            caption_filename = os.path.splitext(image_filename)[0] + '.txt'
            image_path = os.path.join(directory+image_filename)
            caption_path = os.path.join(directory+caption_filename)
            yield (image_path, caption_path if os.path.exists(caption_path) else None)


def copy_captioned_image(image_caption_pair: tuple[str, Optional[str]], source_root: str, target_root: str):
    image_path = image_caption_pair[0]
    caption_path = image_caption_pair[1]

    # make target folder if necessary
    relative_folder = os.path.dirname(os.path.relpath(image_path, source_root))
    target_folder = os.path.join(target_root, relative_folder)
    os.makedirs(target_folder, exist_ok=True)

    # copy files
    shutil.copy2(image_path, os.path.join(target_folder, os.path.basename(image_path)))
    if caption_path is not None:
        shutil.copy2(caption_path, os.path.join(target_folder, os.path.basename(caption_path)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=str, help='Source root folder containing images')
    parser.add_argument('--train_output_folder', type=str, required=True, help="Output folder for the 'train' dataset")
    parser.add_argument('--val_output_folder', type=str, required=True, help="Output folder for the 'val' dataset")
    parser.add_argument('--split_proportion', type=float, required=True, help="Proportion of images to use for 'val' (a number between 0 and 1)")
    parser.add_argument('--seed', type=int, required=False, default=555, help='Random seed for shuffling')
    args = parser.parse_args()

    images = gather_captioned_images(args.source_root)
    print(f"Found {len(images)} captioned images in {args.source_root}")
    val_split_count = math.ceil(len(images) * args.split_proportion)
    if val_split_count == 0:
        raise ValueError(f"No images in validation split with source count {len(images)} and split proportion {args.split_proportion}")

    random.seed(args.seed)
    random.shuffle(images)
    val_split = images[0:val_split_count]
    train_split = images[val_split_count:]
    print(f"Split to 'train' set with {len(train_split)} images and 'val' set with {len(val_split)}")
    print(f"Copying 'val' set to {args.val_output_folder}...")
    for v in tqdm(val_split):
        copy_captioned_image(v, args.source_root, args.val_output_folder)
    print(f"Copying 'train' set to {args.train_output_folder}...")
    for v in tqdm(train_split):
        copy_captioned_image(v, args.source_root, args.train_output_folder)
    print("Done.")

