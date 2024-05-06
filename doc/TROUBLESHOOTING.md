### UserWarning: None of the inputs have requires_grad=True. Gradients will be None

Not an error, this happens during the first time a set of samples are generated.  Ignore.

### CUDA out of memory

See [VRAM](VRAM.md) for more info. 

## ** Some images are smaller than the target size, consider using larger images
## ** Check logs\project_abc_sd21_20230301-122543\undersized_images.txt for more information.

Check the file linked, this is a warning that some of the imags are smaller than the resolution you're using to train.  

The best option is to source the originals again at a higher size. Or you can ignore it, use a high quality upscaler,  remove the images, or reduce the `resolution` you're training.  You might be ok if it is very small percentage of your dataset, but it is something you should check on.

### Errors with PIL

Run this script to check images if your training crashes and PIL or you get any errors that seems image related:

`python scripts/check_images.py --data_root "C:\my_training_data"`

This will help identify invalid or malformatted images.  You can then try using a typical image editor (Photoshop, Gimp, Paint.net, etc.) to fix them, or simply delete them.