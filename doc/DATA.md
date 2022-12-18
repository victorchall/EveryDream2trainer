# Data organization

Since this trainer relies on having captions for your training images you will need to decide how you want deal with this.

There are two currently supported methods to retrieve captions:

1. Name the files with the caption.  Underscore marks the end of the captoin (ex. "john smith on a boat_999.jpg")
2. Put your captions for each image in a .txt file with the same name as the image.  All UTF-8 text is supported with no reserved or special case characters. (ex. 00001.jpg, 00001.txt)

You will need to place all your images and captions into a folder.  Inside that folder, you can use subfolders to organize data as you please.  The trainer will recursively search for images and captions.  It may be useful, for instance, to split each character into a subfolder, and have other subfolders for cityscapes, etc.

When you train, you will use "--data_root" to point to the root folder of your data.  All images in that folder and its subfolders will be used for training.

# Data preparation

## Image size

The trainer will automatically fit your images to the best possible size. It is best to leave your images larger tham you may think for typical Stable Diffusion training.  Even 4K images will be handled fine so just don't sweat it if you have large images.  The only downside is they take a bit more disk space. 

Current recommendation is 1 megapixel (ex 1100x100, 1300x900, etc) or larger, but thinking ahead to future technology advancements you may wish to keep them at even larger resolutions. Again, don't worry about the trainer squeezing or cropping, it will handle it!

Aspect ratios up to 4:1 or 1:4 are supported.  Again, just don't worry about this too much.  The trainer will handle it.

## Cropping

You can crop your images in an image editor *if you need, in order to get good close ups of things like faces, or to split images up that contain multiple characters.*  As above, make sure **after** cropping your images are still fairly large.  It is ok to use a full shot of two characters in one image and also a cropped version of each character separately, but make sure every image is captioned appropriately for what is actually present in each image.

## Captions

For most use cases, use a sane English sentence to describe the image.  Try to put your character or main object name close to the start.

### Styles

For style, consider adding a suffix on the caption that describes the style.  Examples would be "by claude monet" or "in the style of gta box art" at the end of the caption.  This will help the model learn recall style at inference time so you can style other subjects you did not train with the style. You may also consider "drawing of" or "painting of" at the start of the caption when appropriate.

Consider also including a style tag as above if you are training anything besides photos.  For instance, if you are training a few characters from a video game you can consider "cloud strife holding a buster sword, screenshot from final fantasy for ps5" if you wish to capture the style of the game along with the characters.

### Context

Include the surroundings and context in your captions.  Ex. "cloud strife standing on a dirt path in midgar city slums district"  Again, this will allow you to recall the "dirt path in midgar city slums district" style at inference time, and will even pick up on pieces of that like "midgar city" (if enough samples are present with similar words) as a style you can apply later!

Also consider some basic mention of pose.  ex. "clouds strife sitting on a blue wooden bench in front of a concrete wall" or "barrett wallace holding his fist in front of his face with an angry look on his face, looking at the camera."  Captions can capture value not only for the character's look, but also for the pose, the background scene, and the camera angle.  You can be creative here, there is a lot of potential!