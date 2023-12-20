# Plugin support

This is a very early and evolving feature, but users who have a need to extend behavior can now do so with plugin loading.

## Plugin creation

To create a plugin, extend the BasePlugin class and override the methods you want to change.  

For example, let's say we made the `ExampleLoggingPlugin` class, and placed it in a file called `plugins/example_plugin.py`:

To activate a plugin, edit your `train.json` as follows:

```"plugins": ["plugins.example_plugin.ExampleLoggingPlugin"],```

The entries are in 'module-path-like' form to your class. For this example, `plugins/example_plugin.py` contains a class `ExampleLoggingPlugin` which extends `BasePlugin`. See this [example](../plugins/example_plugin.py) for more details.

You can add as many plugins as you want, and each plugin only needs to implement the functions you wish.

Data is passed to the plugin functions as a list of variables via `kwargs`, but it's still an open question which data should be passed in. In any case it should be relevant to the function, e.g. `on_epoch_start()` should not receive batch data or anything specific to a step, except for `epoch` and `global_step`. 

## Quick example

A basic implementation of a plugin that only uses the "end of an epoch" function will look something like this:

```python
from plugins.plugins import BasePlugin

class MyPlugin(BasePlugin):
    def on_epoch_end(self, **kwargs):
        print(f"hello world, this is a plugin at the end of epoch number: {kwargs['epoch']}")
```

## Data loader hooks

These runs every time the image/caption pair are loaded and passed into the training loop.  These do not use kwargs, but are focused on specific parts of the data loading routine. 

#### transform_caption(self, caption:str)
Could be useful for things like customized shuffling algorithms, word replacement/addition/removal, randomization of length, etc. 

#### transform_pil_image(self, img:Image)
Could be  useful for things like color grading, gamma adjustment, HSL modifications, etc.  Note that AFTER this function runs the image is converted to numpy format and normalized (std_dev=0.5, norm=0.5) per the reference implementation in Stable Diffusion, so normalization is wasted compute. From prior experimentation, all adjustments to this normalization scheme degrade output of the model, thus are a waste of time and have been hardcoded.  Gamma or curve adjustments are still potentially useful, as are hue and saturation changes. 
