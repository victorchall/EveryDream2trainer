# Captioning tools

## Open-Flamingo

`python caption_fl.py --data_root input --min_new_tokens 20 --max_new_tokens 30 --num_beams 3 --model "openflamingo/OpenFlamingo-9B-vitl-mpt7b"`

This script uses two example image/caption pairs located in the example folder to prime the system to caption, then captions the images in the input folder. It will save a `.txt` file of the same base filename with the captoin in the same folder. 

This script currently requires an AMPERE or newer GPU due to using bfloat16. 

**Trying out different example image/caption pairs will influence how the system captions the input images.** Adding more examples slows processing. 

Supported models:

* `openflamingo/OpenFlamingo-3B-vitl-mpt1b` Small model, requires 8 GB VRAM a num_beams 3, or 12 GB at num_beams 16
* `openflamingo/OpenFlamingo-9B-vitl-mpt7b` Large model, requires 24 GB VRAM at num_beams 3, or 36.7gb at num_beams 32

The small model with more beams (ex. 16) performs well with details and should not be immediately discounted. 

The larger model is more accurate with proper names (i.e. identifying well-known celebrities, objects, or locations) and seems to exhibit a larger vocabulary.

Primary params:

* `--num_beams 3` increasing uses more VRAM and runs slower, may improve detail, but can increase hallicunations
* `--min_new_tokens 20` and `--max_new_tokens 35` control the length of the caption

Other settings:

* `--force_cpu` forces to use CPU even if a CUDA device is present
* `--temperature 1.0` relates to randomness used for next token chosen
* `--repetition_penalty 1.0` penalizes repeating tokens/words, can adjust up if you see repeated terms
* `--length_penalty 1.0` penalizes longer captions
