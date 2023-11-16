import os
import torch
import safetensors as st
from safetensors.torch import save_file
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    args = argparser.parse_args()

    print(f"Loading model {args.ckpt}")
    model = torch.load(f"{args.ckpt}")
    print("Model loaded.")
    base_name = os.path.splitext(args.ckpt)[0]
    print(f"base_name: {base_name}")

    model = model.pop("state_dict", model)
    save_file(model, f"{base_name}.safetensors")
    print(f"File converted to safetensors:  {base_name}.safetensors")