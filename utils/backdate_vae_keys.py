import os
import torch
from safetensors.torch import save_file, load_file

reshapes = ["first_stage_model.decoder.mid.attn_1.to_k.weight",
            "first_stage_model.decoder.mid.attn_1.to_q.weight",
            "first_stage_model.decoder.mid.attn_1.to_v.weight",
            "first_stage_model.encoder.mid.attn_1.to_k.weight",
            "first_stage_model.encoder.mid.attn_1.to_q.weight",
            "first_stage_model.encoder.mid.attn_1.to_v.weight",
            "first_stage_model.decoder.mid.attn_1.to_out.0.weight",
            "first_stage_model.encoder.mid.attn_1.to_out.0.weight"
            ]

def _reshape(state_dict, key):    
    if key in reshapes:
        if state_dict[key].dim() == 2:
            old_shape = state_dict[key].shape
            # add two dimensions after last dim
            state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)
            print(f" ** reshaped {key} from {old_shape} to {state_dict[key].shape}")
        else:
            print(f" ** skipping {key} because it is already correct shape {state_dict[key].shape}")

def fix_vae_keys(state_dict, is_sd1=True):
    if not is_sd1:
        return state_dict

    new_state_dict = {}

    with open("backdate_vae_keys.log", "w") as f:
        f.write(f"keys:\n")
        changed_i = 0

        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' not in state_dict:
            # openai clip-l for some reason has this defined as part of its state_dict, which is dumb, but whatever
            state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = torch.linspace(0, 76, 77, dtype=torch.int64).unsqueeze(0)

        for key in state_dict.keys():
            new_key = key
            _reshape(state_dict, key)
            if key.startswith("first_stage_model"):
                
                if ".to_q" in key:
                    print(f" *  backdating {key} {state_dict[key].shape}")
                    new_key = new_key.replace('.to_q.', '.q.')
                    print(f" ** new key -> {new_key}\n")
                elif ".to_k" in key:
                    print(f" *  backdating {key} {state_dict[key].shape}")
                    new_key = new_key.replace('.to_k.', '.k.')
                    print(f" ** new key -> {new_key}\n")
                elif ".to_v" in key:
                    print(f" *  backdating {key} {state_dict[key].shape}")
                    new_key = new_key.replace('.to_v.', '.v.')
                    print(f" ** new key -> {new_key}\n")
                elif ".to_out.0" in key:
                    print(f" *  backdating {key} {state_dict[key].shape}")
                    new_key = new_key.replace('.to_out.0', '.proj_out')
                    print(f" ** new key -> {new_key} {state_dict[key].shape}\n")

            new_state_dict[new_key] = state_dict[key]

            changed = 1 if key != new_key else 0
            f.write(f"{changed}: {key} -- {new_key} {new_state_dict[new_key].shape}\n")

    return new_state_dict

def _backdate_keys(filepath, state_dict):
    new_state_dict = fix_vae_keys(state_dict)
    base_path_without_ext = os.path.splitext(filepath)[0]
    ext = os.path.splitext(filepath)[1]

    new_path = f"{base_path_without_ext}_fixed{ext}"
    print (f"Saving to {new_path}")
    save_file(new_state_dict, new_path)

def _compare_keys(filea_state_dict, fileb_state_dict):
    #remove cond_stage_model.transformer.text_model.embeddings.position_ids key
    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in filea_state_dict:
        filea_state_dict.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)
    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in fileb_state_dict:
        fileb_state_dict.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)

    # sort the keys for comparison (best shot we have at a fair comparison without trying to count params and other nonsense)
    filea_state_dict_keys = sorted(filea_state_dict.keys())
    fileb_state_dict_keys = sorted(fileb_state_dict.keys())

    print("filea keys         <----->          fileb keys")
    # compare the keys line by line
    for filea_key, fileb_key in zip(filea_state_dict_keys, fileb_state_dict_keys):
        if filea_key != fileb_key:
            print("Mismatched keys:")
            print (f"   filea key: {filea_key} {filea_state_dict[filea_key].shape}")
            print (f"   fileb key: {fileb_key} {fileb_state_dict[fileb_key].shape}")            
        else:
            #print (f"{ckpt_key} == {st_key}")
            pass
    print("filea keys         <----->          fileb keys")

def _load(filepath):
    if filepath.endswith(".safetensors"):
        print(f" Loading {filepath} loading as safetensors file")
        state_dict = load_file(filepath)
    else: # LDM ckpt
        print(f" Loading {filepath} loading as LDM checkpoint")
        state_dict = torch.load(filepath, map_location='cpu')['state_dict']
    return state_dict

def _dump_keys(filepath, state_dict):
    with open(filepath, "w") as f:
        state_dict_keys = sorted(state_dict.keys())
        for key in state_dict_keys:
            f.write(f"{key} - {state_dict[key].shape}\n")

if __name__ == "__main__":
    print("BACKDATE AutoencoderKL/VAE KEYS TO OLD NAMES SCRIPT OF DOOM")
    print("================================")
    print(" --filea <path to ckpt or safetensors file>    file to backdate the VAE keys (will make a copy as <filename>_fixed.safetensors)")
    print(" --fileb <path to ckpt or safetensors file>    to compare keys to filea")
    print(" --compare                                     to run keys comparison (requires both --filea and --fileb)")
    print(" --backdate                                    to backdate keys (only for --filea)")
    print(" --dumpkeys                                    to write key and shapes for either or both files keys for files to '<filename>.txt'")
    print(" You must specify one of --compare or --backdate or --dumpkeys to do anything.")
    print(" ex.   python utils/backdate_vae_keys.py --filea my_finetune.safetensors --fileb original_sd15.ckpt --compare")
    print(" ex.   python utils/backdate_vae_keys.py --filea my_finetune.safetensors --backdate")
    print(" ex.   python utils/backdate_vae_keys.py --filea what_is_this_model_shape.safetensors --dumpkeys")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filea", type=str, required=True, help="Path to the safetensors file to fix")
    parser.add_argument("--fileb", type=str, required=False, help="Path to the safetensors file to fix")
    parser.add_argument("--compare", action="store_true", help="Compare keys")
    parser.add_argument("--backdate", action="store_true", help="backdates the keys in filea only")
    parser.add_argument("--dumpkeys", action="store_true", help="dump keys to txt file")
    args = parser.parse_args()

    filea_state_dict = _load(args.filea) if args.filea else None
    fileb_state_dict = _load(args.fileb) if args.fileb else None

    if args.dumpkeys:
        print(f"Dumping keys to txt files")
        if args.filea:
            _dump_keys(f"{os.path.splitext(args.filea)[0]}.txt", filea_state_dict)
        if args.fileb:
            _dump_keys(f"{os.path.splitext(args.fileb)[0]}.txt", fileb_state_dict)
    if args.compare and not args.backdate:
        print(f"Comparing keys in {args.filea} to {args.fileb}")
        _compare_keys(filea_state_dict, fileb_state_dict)
    elif args.backdate:
        print(f"Backdating keys in {args.filea}")
        print(f"   ** ignoring {args.fileb}") if args.fileb else None
        _backdate_keys(args.filea, filea_state_dict)
    else:
        print("Please specify only --compare with both --filea and --fileb to compare keys and print differences to console")
        print(" ... or --backdate with only --filea to backdate its keys to old LDM names and save to <filea>_fixed.safetensors")
