# Shuffling tags randomly during training

## General shuffling

To help the model generalize better, EveryDream has an option to shuffle tags during the training.

This behavior can be activated using the parameter _--shuffle_tags_. The default is off.

The provided caption, extracted either from the file name or the provided caption file, 
will be split at each "_,_" into separate chunks.     

The first chunk will always be included in the caption provided during the training, 
the additional chunks are shuffled into a random order. 

Each epoch the order is reshuffled. _(Remember that each image is shown one per epoch to the model)_


## Weighted shuffling

EveryDream can read caption definitions from YAML files, for fine-tuned definitions.

EveryDream will check for each image if a file with the same name and the extension _.yaml_ is provided.

The expected format of the YAML file:
````yaml
main_prompt: A portrait of Cloud Strife
tags:
  - tag: low angle shot
  - tag: looking to the side
  - tag: holding buster sword
    weight: 1.5
  - tag: clouds in background
    weight: 0.5
  - tag: smiling
    weight: 0.8
max_caption_length: 1024
````

THe main prompt will always be the first part included in the caption.
The main prompt is optional, you can provide none if you do not want a fixed part at the beginning of the caption.

This is followed by a list of tags. The tags will be shuffled into a random order and added to the caption.
The tags list is optional.   

The default weight of each tag is _1.0_. A different weight can be optionally specified. 
Tags with a higher weight have a higher chance to appear in the front of the caption tag list.

The optional parameter _max_caption_length_ allows the definition of a maximum length of the assembled caption.
Only whole tags will be processed. If the addition of the next tag exceeds the _max_caption_length_, 
it will not be added, and the caption will be provided without the other tags for this epoch.  

This can be used to train the model that an image can include a certain aspect, even if it is not 
explicitly mentioned in the caption. 


## General notes regarding token length

For SD, the current implementation of EveryDream can only process the first 75 tokens 
provided in the caption during training.

This is a base limitation of the SD Models. Workaround exists to extend this number but are currently not
implemented in EveryDream.

The effect of the limit is that the caption will always be truncated when the maximum number of tokens is
exceeded. This process does not consider if the cutoff is in the middle of a tag or even in the middle of a
word if it is translated into several tokens.

