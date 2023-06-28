import argparse
import importlib
import logging

class BasePlugin:
    def on_epoch_start(self, **kwargs):
        pass
    def on_epoch_end(self, **kwargs):
        pass

class ExampleLoggingPlugin(BasePlugin):
    def on_epoch_start(self, **kwargs):
        logging.info(f"Epoch {kwargs['epoch']} starting")
    def on_epoch_end(self, **kwargs):
        logging.info(f"Epoch {kwargs['epoch']} finished")

def load_plugin(plugin_name):
    module = importlib.import_module(plugin_name)
    plugin_class = getattr(module, plugin_name)
    if not issubclass(plugin_class, BasePlugin):
        raise TypeError(f'{plugin_name} is not a valid plugin')
    logging.info(f"Plugin {plugin_name} loaded")
    return plugin_class()
