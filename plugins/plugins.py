import argparse
import importlib
import logging
import time
import warnings
from PIL import Image

class BasePlugin:
    def on_epoch_start(self, **kwargs):
        pass
    def on_epoch_end(self, **kwargs):
        pass
    def on_training_start(self, **kwargs):
        pass
    def on_training_end(self, **kwargs):
        pass
    def on_step_start(self, **kwargs):
        pass
    def on_step_end(self, **kwargs):
        pass
    def on_will_step_optimizer(self, **kwargs):
        pass
    def transform_caption(self, caption:str) -> str:
        return caption
    def transform_pil_image(self, img:Image) -> Image:
        return img
    def modify_sample_prompt(self, prompt:str) -> str:
        return prompt

def load_plugin(plugin_path):
    print(f" - Attempting to load plugin: {plugin_path}")
    module_path = '.'.join(plugin_path.split('.')[:-1])
    module = importlib.import_module(module_path)
    plugin_name = plugin_path.split('.')[-1]

    plugin_class = getattr(module, plugin_name)

    if not issubclass(plugin_class, BasePlugin):
        raise TypeError(f'{plugin_path} is not a valid plugin')
    logging.info(f" - Plugin {plugin_path} loaded to {plugin_class}")
    return plugin_class()

class Timer:
    def __init__(self, warn_seconds, label='plugin'):
        self.warn_seconds = warn_seconds
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self.start
        if elapsed_time > self.warn_seconds:
            logging.warning(f'Execution of {self.label} took {elapsed_time} seconds which is longer than the limit of {self.warn_seconds} seconds')


class PluginRunner:
    def __init__(self, plugins:list=[], epoch_warn_seconds=5, step_warn_seconds=0.5, training_warn_seconds=20):
        """
        plugins: list of plugins to run
        epoch_warn_seconds: warn if any epoch start/end call takes longer than this
        step_warn_seconds: warn if any step start/end call takes longer than this
        training_warn_seconds: warn if any training start/end call take longer than this
        """
        self.plugins = plugins
        self.epoch_warn_seconds = epoch_warn_seconds
        self.step_warn_seconds = step_warn_seconds
        self.training_warn_seconds = training_warn_seconds

    def run_on_epoch_end(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.epoch_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_epoch_end(**kwargs)

    def run_on_epoch_start(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.epoch_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_epoch_start(**kwargs)

    def run_on_training_start(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.training_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_training_start(**kwargs)

    def run_on_training_end(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.training_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_training_end(**kwargs)

    def run_on_step_start(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.step_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_step_start(**kwargs)

    def run_on_step_end(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.step_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_step_end(**kwargs)

    def run_on_backpropagation(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.step_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_backpropagation(**kwargs)

    def run_on_model_load(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.training_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_model_load(**kwargs)

    def run_on_model_save(self, **kwargs):
        for plugin in self.plugins:
            with Timer(warn_seconds=self.training_warn_seconds, label=f'{plugin.__class__.__name__}'):
                plugin.on_model_save(**kwargs)

    def run_transform_caption(self, caption):
        with Timer(warn_seconds=self.step_warn_seconds, label="plugin.transform_caption"):
            for plugin in self.plugins:
                caption = plugin.transform_caption(caption)
        return caption

    def run_transform_pil_image(self, img):
        with Timer(warn_seconds=self.step_warn_seconds, label="plugin.transform_pil_image"):
            for plugin in self.plugins:
                img = plugin.transform_pil_image(img)
        return img

    def run_modify_sample_prompt(self, prompt) -> str:
        with Timer(warn_seconds=self.step_warn_seconds, label="plugin.modify_sample_prompt"):
            for plugin in self.plugins:
                prompt = plugin.modify_sample_prompt(prompt)
        return prompt
