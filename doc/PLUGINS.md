# Plugin support

This is a very early and evolving feature, but users who have a need to extend behavior can now do so with plugin loading.

To create a plugin, extend the BasePlugin class and override the methods you want to change.  

For example, lets say we made ExampleLoggingPlugin class, and placed it in a file called `plugins/example_plugin.py`:


To activate a plugin, edit your `train.json` and use 

```"plugins": ["plugins.example_plugin.ExampleLoggingPlugin"],```

This uses module-path-like to your class.  For this example, `plugins/example_plug.py` contains a class `ExampleLoggingPlugin` which extends `BasePlugin`. You can go look at the [example](plugins/example_plugin.py) to see how this one works.

Everything is passed in as kwargs and the list of variables passed in is not fully defined yet.  Consider it an open question what should be passed in, but it should be specific to the context, such as on_epoch_start should not pass in batch data or anything specific to step except for `epoch` and `global_step`. 

You can pass in as many plugins as you want, and each plugin only needs to implement the functions you wish.

A basic implementation of a plugin that only uses the "end of an epoch" function will look something like this:

```python
from plugins.plugins import BasePlugin

class MyPlugin(BasePlugin):
    def on_epoch_end(self, **kwargs):
        print(f"hello world, this is a plugin at the end of epoch number: {kwargs['epoch']}")
```