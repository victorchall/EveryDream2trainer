# Selecting and preparing images

## Number of images

You should probably start with less than 100 images, until you get a feel for training. When you are ready, ED2 supports up to tens of thousands of images.

## Image size and quality
ED2 supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, and `.jfif` image formats.

Current recommendation is _at least_ 1 megapixel (ex 1024x1024, 1100x900, 1300x800, etc). That being said, technology continues to advance rapidly. ED2 has no problem handling 4K images, so it's up to you to pick the appropriate trade-off with disk and network costs. 


Scaling images up is not a great idea though, though it may be tolerable as a very small percentage of your data set. If you only have 512x512 images, don't try training at 768.

Use high quality, in-focus, low-noise, images, capturing the concept(s) under training with high fidelity wherever possible. 

## Cropping and Aspect Ratios

You can crop your images in an image editor __if it highlights the concept under training__, e.g. to get good close ups of things like faces, or to split images up that contain multiple characters.  

**You do not need to crop to a square image**

Aspect ratios between 4:1 and 1:4 are supported; the trainer will handle bucketing and resizing your images as needed. 

It is ok to use a full shot of two characters in one image and also a cropped version of each character separately, but make sure every image is captioned appropriately for what is actually present in each image.


## Caption Design

A caption consists of a main prompt, followed by one or more comma-separated tags.  

For most use cases, use a sane English sentence to describe the image.  Try to put your character or main object name close to the start.

**If you are training on images of humans, there is little benefit in using "unique" names most of the time**. Don't worry so much about using a "rare" toking, or making up gibberish words. Just try generating a few images using your concept names, and make sure there are no serious conflicts.

Those training anime models can use booru tags as well using other utilities to generate the captions.

### Styles

For style, consider adding a suffix on the caption that describes the style.  Examples would be "by claude monet" or "in the style of gta box art" at the end of the caption.  This will help the model learn recall style at inference time so you can style other subjects you did not train with the style. You may also consider "drawing of" or "painting of" at the start of the caption when appropriate.

Consider also including a style tag as above if you are training anything besides photos.  For instance, if you are training a few characters from a video game you can consider "cloud strife holding a buster sword, screenshot from final fantasy for ps5" if you wish to capture the "style" of the game render along with the characters.

### Context

Include the surroundings and context in your captions.  Ex. "cloud strife standing on a dirt path in midgar city slums district"  Again, this will allow you to recall the "dirt path in midgar city slums district" style at inference time, and will even pick up on pieces of that like "midgar city" (if enough samples are present with similar words) as a style or scenery you can apply later.  This can extract additional value from your training besides just the character.

Also consider some basic mention of pose.  ex. "clouds strife sitting on a blue wooden bench in front of a concrete wall" or "barrett wallace holding his fist in front of his face with an angry look on his face, looking at the camera."  Captions can capture value not only for the character's look, but also for the pose, the background scene, and the camera angle.  You can be creative here, there is a lot of potential!


# Constructing a dataset
A dataset consists of image files coupled to captions and other configuration.

You are welcome to use any folder structure that makes sense for your project, but you should know that there are configuration tricks that rely on data being partitioned into folders and subfolders.

## Assigning captions
### by Filename
The simplest and least powerful way to caption images is by filename. The name of the file, without extension, and excluding any characters after an _ (underscore). 

```
a photo of ted bennet, sitting on a green armchair_1.jpg
a photo of ted bennet, laying down_1.jpg
a photo of ted bennet, laying down_2.jpg
```
### by Caption file
If you are running in a Windows environment, you may not be able to fit your whole caption in the file name.

Instead you can create a text file with the same name as your image file, but with a `.txt` or `.caption` extension, and the content of the text file will be used as the caption, **ignoring the name of the file**.

### by Caption yaml
You can capture a more complex caption structure by using a `.yaml` sidecar instead of a text file. Specifically you can assign weights to tags for [shuffling](SHUFFLING_TAGS.md).

The format for `.yaml` captions:
```
main_prompt: a photo of ted bennet
tags:
  - "sitting on a green armchair" # The tag can be a simple string
  - tag: "wearing a tuxedo" # or it can be a tag string
    weight: 1.5             # optionally paired with a shuffle weight
```


### Assigning captions to entire folders
As mentioned above, a caption is a main prompt accompanied by zero or more tags. 
Currently it is not possible for a caption to have more than one main tag, although this limitation may be removed in the future.

But, in some cases it may make sense to add the same tag to all images in a folder. You can place any configuration that should apply to all images in a local folder (ignoring anything in any subfolders) by adding a file called `local.yaml` to the folder. In this file you can, for example, add:
```
tags:
  - tag: "in the style of xyz"
```
And this tag will be appended to any tags specified at the image level.

If you want this tag, or any other configuration, to to apply to images in subfolders as well you can create a file called `global.yaml` and it will apply to all images in the local folder **and** to any images in any subfolders, recursively.

## Other image configuration
In addition to captions, you can also specify the frequency with which each image should show up in training (`multiply`), or the frequency in which the trainer should be given a flipped version of the image (`flip_p`), or the frequency in which the caption should be dropped completely focusing the training on the image alone, ignoring the caption (`cond_dropout`).

For simple cases you can create a file called `multiply.txt`, `flip_p.txt`, and/or `cond_dropout.txt`, containing the single numeric value for that configuration parameter that should be applied to all images in the local folder.

Alternatively you can add these properties to any of the supported `.yaml` configuration files, image-level, `local.yaml`, and/or `global.yaml`

```
main_prompt: a photo of ted bennet
tags:
  - sitting on a green armchair
multiply: 2
flip_p: 0.5
cond_droput: 0.05
```

See [Advanced Tweaking](ATWEAKING.md) for more information on image flipping and conditional dropout.

The [Data Balancing](BALANCING.md) guide has some more information on how to balance your data using multipliers, and what to consider for model preservation and mixing in ground truth data.
