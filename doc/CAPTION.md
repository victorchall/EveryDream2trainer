# Older Captioning tools

## Kosmos-2

Microsoft's [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)  is significantly lighter weight than Cog, using <5GB of VRAM and generating captions in under a second on a RTX 3090.  

It has the capability to output grounding bounding boxes.

Run `python caption_kosmos2.py --help` to get a list of options. 

You can use `--prompt` to provide a prompt.  The official suggested prompts are `An image of` or `Describe this image in detail:`.  The later is the default if you do not set a prompt.
If you want to use Kosmos-2 as a VQA (visual question answering), format your prompt like so `Question: Is there watermark on this image? Answer:`.

### _Kosmos-2 grounding_

Kosmos-2can generate bounding boxes for the "grounding" of the caption.  This is useful for identifying specific objects in the image in 2D space, which can be useful in later piplines. 

It's worth reading the documentation [here](https://huggingface.co/microsoft/kosmos-2-patch14-224) to understand the grounding output.

`--save_entities` outputs a '.ent' file with bounding box information.  The entities identified will be based on what caption is produced.

`--phrase_mode` This modifies how the model is called, wrapping phrases in \<phrase> tags to identify specific classes.  This also interprets your prompt as a CSV, wrapping each item in a phrase tag. You might use it with `--prompt "dog,cat,tree"` for instance.  *This is not a gaurantee your phrases will be found and output into the grounding output file.* Things like  `--phrase_mode --prompt "watermark"` might work as a poor man's watermark detector, but with mixed results so its best to test with your data.

`--save_entities_only` This will not attempt to write the caption into the .txt file at all.  **This is recommended with `--phrase_mode` for object detection**. Using this option forces `--save_entities`.

There is a trivial/dumb UI for viewing the grounding in the scripts folder.  Launch it with `python scripts/grounding_ui.py` and it will open a window allowing you to select a directory, and it will display the images and bounding boxes. 

## Blip, Blip2, and git

Older script:

`caption_blipgit.py`
