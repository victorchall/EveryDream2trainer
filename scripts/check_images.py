import os
import argparse
from PIL import Image, ImageOps
from typing import Generator

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def image_path_generator(image_dir: str, do_recurse: bool = True) -> Generator[str, None, None]:
    if do_recurse:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                    yield os.path.join(root, file)
    else:
        for file in os.listdir(image_dir):
            if any(file.endswith(ext) for ext in SUPPORTED_EXT):
                yield os.path.join(image_dir, file)

def main(args):
    failed = []
    for path in image_path_generator(args.data_root, do_recurse=True):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            print(f"Checked OK {img.size} {img.mode} {path}")
        except Exception as e:
            print(f"FAILED: {path}")
            failed.append((path,e))

    if not failed:
        print("No errors found")
    else:
        print(f" *************** Errors were found ***************")
        for path, e in failed:
            print(f"FAILED: {path} {e}")

if __name__ == '__main__':
    print("This script checks that all images in a directory are valid.")
    print("If any errors occur, they will be printed out at the end.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to the root of the dataset to check')
    args = parser.parse_args()
    main(args)
