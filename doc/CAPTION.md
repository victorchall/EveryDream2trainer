# Captioning tools

## Open-Flamingo

`python caption_fl.py --data_root input --min_new_tokens 20 --max_new_tokens 30 --num_beams 3 --model "openflamingo/OpenFlamingo-9B-vitl-mpt7b"`

`--num_beams 3` increasing uses more VRAM but may improve detail, also can increase hallicunations
`--min_new_tokens 20` and `--max_new_tokens 35` control the length of the caption

Other settings:
`--force_cpu` forces to use CPU even if a CUDA device is present
`--temperature 1.0` relates to randomness used for next token chosen
`--repetition_penalty 1.0` penalizes repeating tokens/words, can adjust up if you see repeated terms
`--length_penalty 1.0` penalizes longer captions
