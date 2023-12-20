# Plugin support

This is a very early and evolving feature, but users who have a need to extend behavior can now do so with plugin loading and without having to edit the main training software.

This allows developers to experiment without having to manage branches, or maintain very custom or narrow-use-case behaviors that are not appropriate with which to clutter the primary software. 

Not everything is necessarily possible or convenient with this plugin system, but it should handle a substantial number of experimental, unproven, or narrow-use-case-specific functionality.  For instance, one could invent nearly an infinite number of different ways to shuffle captions, but adding dozens of arguments to the main training script for these is simply inappropriate and leads to cluttered code and user confusion.  These instead should be implemented as plugins. 

Plugins are also a good entry point to people wanting to get their feet wet making changes.  Often the context is small enough that a tool like ChatGPT or your own local LLM can write these for you if you can write reasonable requirements. 

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

## Adding hooks

Additional hooks may be added to the core trainer to allow plugins to be run at certain points in training or to transform certain things during training.  Typically the plugin running itself is not a performance concern so adding hooks by itself is not going to cause problems.

PluginRunner is the class that loads and manages all the loaded plugins and calls the hook for each of them at runtime. The `plugin_runner` instance of this class is created in the main trainer script, and you may need to inject it elsewhere depending on what context is required for your hook exeuction. 

To add a new hook:
1. Edit BasePlugin class to add the hook function.  Define the function and implemenet it as a no-op using `pass` or simply returning the thing to be transformed with no transformation.
2. Edit PluginRunner class to add the function that will loop all the plugins and call the hook function defined in step 1.
3. Edit the main training software to call `plugin_runner.your_runner_loop_fn(...)` as defined in step 2. 
