# Data Balancing and Preservation using multiply.txt

One of the challenges in building a dataset when you wish to train many concepts or subjects is balancing the number of images per concept or subject.  For instance, you may be able to source 200 images of one subject or style, and only 50 of another.  This can lead to a model that is biased towards the subject with more images.  To combat this, you can use the `multiply.txt` file to "virtually duplicate" images in the dataset, **OR** reduce the number of times images are repeated.

## Balancing your subjects

In the above example with a mismatch dataset of 200 images of Subject A and 50 of Subject B, you can place `multiply.txt` in the subfolder for Subject B simply with the number 4 in it.  This will cause the dataset to repeat each image 4 times, bringing the total number of images for Subject B per epoch 200.

## What if I don't use multiply.txt

The default behavior is for the trainer to repeat each image in your `data_root` folder, recursively (all subfolders), 1 time per epoch.  

Keep in mind, however, `multiply.txt` is *not* recursively applied to subfolders.  Keep reading for more details.

## How exactly do I use multiply.txt?

Place a text file called `multiply.txt` in the folder with the images you wish to affect.  Only the specific folder is affected, as this does not "cascade" down to any subfolders.  The value can be any real number greater than zero, such as `2` or `0.5`.  My general suggestion is to stick with numbers in the 1-3 range for data balancing purposes, and use numbers <1 for other purposes (see below).

Values of whole numbers make sure the images in that folder are present in each epoch exactly that number of times. 

Any fraction (ex `0.4`) or leftover fraction (ex. `2.5` has a leftover of `0.5`) mean that the remaining portion is randomly selected every epoch from that folder.  For instance, if you have 100 images in a folder and you set `multiply.txt` to `2.5`, all images are repeated twice (100>200), then 50 images of the 100 (the `0.5` remainder) will be selected randomly every epoch.

## How do I know I did it right?

The trainer will dump a message in your log every time it finds a `multiply.txt` file and applies it, so you can confirm it is working by looking at your log or your console screen.  This happens during the preloading phase of the trainer one time per training session and does not happen every epoch.

## I'm confused, what is this really doing?

You can use the `write_schedule` option to turn on an output of the images every epoch in your logs.  Try running 1 epoch, then go change your `multiply.txt` in one of your data folders, then run 1 epoch again.  Compare the `ep0_batch_schedule.txt` files between the two runs.  You will see the images are repeated or not repeated as you expect.

As as above, check your .log file or console screen to make sure the `multiply.txt` file(s) are seen by the training during preloading.  If you accidentally misname the file, such as `multiply.txt.txt` (which can be a common issue on Windows if you have 'hide file extensions` on in File Explorer) it won't work. 

## Do my concepts or subjects really need to be equalized? 

This is actually a great queston.  The answer is, it depends.

Not all subject matter train at the same rate.  For instance, if you train `Person A` and `Person B`, it may be you see in your samples than `Person B` trains faster, *even when you have exactly the same number of images for each person.*  This is because one face may be more "familiar" to Stable Diffusion, maybe because they look more like the average person or some celebrity.  The same could apply to different art styles.

It is hard to know ahead of time, which is where you, the human running the training process, comes in to subjectively judge how your training is going and make adjustments.  Experimentation is part of the training process.

You can use `multiply.txt` to slightly adjust the number of times images are repeated to help balance the training.  In the above example, you could use `multiply.txt` for the `Person A` folder and set it to `1.2` to give `Person A` a 20% boost in repetition.  You could also **instead** put `multiply.txt` in the `Person B` folder and set it to `0.8` to give `Person B` a 20% reduction in repetition.  Either way could work, but I generally suggest you tend to using values >1 for this purpose, and save values <1 for other purposes (see [What is preservation?](#What_is_preservation)).


## What is preservation?

A quick diversion and preamble is needed here before getting into [Using multiply.txt to help with model preservation](#using-multiplytxt-to-help-with-model-preservation) below.

To define the term, when I use the term "preservation" I mean attempts to prevent the model from forgetting what it has learned in the past.  

Since Stable Diffusion already has a wide range of information stored, and because typical training for new subject matter is done by "hammering" the data in over and over (i.e. using many epochs on specific data), this can cause Stable Diffusion to "forget" prior learning, leading to various types of artifacts in the output later after training.  Examples are sunburnt looking faces, malformed faces, or faces that are too smooth if you are training, say, CG characters.  Or, you may find the style of your training images "bleeds" into the rest of the model even when not prompted, like suddenly a prompt simply of "tom cruise" draws him as a cartoon character when you train heavily on cartoon characters.

The Dreambooth implementations out there typically introduce generated images out of the model itself into the training set. I do not suggest that, and that's why EveryDream is not a Dreambooth implementation.  I instead suggest you can gather other alternative "ground truth" data and "mix it in" with your training data to "preserve" the model and avoid "forgetting" or other artifacts.  The tools repo has a Laion data scraper, or you can use things datasets like FFHQ photo set. 

I recommend "ground truth" images as opposed to the Dreambooth implementations which use generated images out of the model itself for preservation.  Dreambooth training on generated images can reinforce the error already present in the model.  Ex. 6 fingers, bad limbs, etc.  

Real ground truth images and webscrapes are much less likely to have 6 fingers or 3 arms, etc.  This is the whole point of moving away from Dreambooth and what sets EveryDream apart from those trainers, as I've focused the code on features than enable this method of fine tuning instead of Dreambooth.

## Using multiply.txt to help with model preservation

Given the above paragraphs on preservation, therefore, another way to utilize `multiply.txt` is to use a fractional number in a large "preservation data set".  

For instance, let's say you have 200 images of a subject you wish to train, and have collected a web scrape of 1000 images of a variety of styles and subjects to help the model remember and avoid "artifacts" and "bleeding" while hammering in your 200 images of a new subject.  If you train in a normal manner, you will be repeating the entire 1200 (1000+200) images at the same rate.  But for preservation, you do not need to repeat the 1000 preservation images so often as it is just there to help the model remember what it already knows.  It's simply not necessary to repeat these at the same rate as your new training images, and will lengthen training time unnecessarily. 

In this case with 200 training images and 1000 preservation images, I would suggest placing `multiply.txt` in the  subfolder with your preservation images with the number around the range of `0.05` to `0.1`.  This will cause training to randomly select 50-100 preservation images (`1000*0.05=50` or `1000*0.10=100`) per epoch, while the actual training images (200) all get trained once per epoch.