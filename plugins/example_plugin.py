from plugins.plugins import BasePlugin
import logging
from colorama import Fore, Style

class ExampleLoggingPlugin(BasePlugin):
    def __init__(self):
        print(f"{Fore.LIGHTBLUE_EX}ExampleLoggingPlugin init{Style.RESET_ALL}")
        # Setup any state variables here
        pass

    def on_epoch_start(self, **kwargs):
        logging.info(f"{Fore.LIGHTBLUE_EX} ** ExampleLoggingPlugin: on_epoch_start{Style.RESET_ALL}")
        for k, v in kwargs.items():
            logging.info(f"  {Fore.BLUE}{k}: {v}{Style.RESET_ALL}")

    def on_epoch_end(self, **kwargs):
        logging.info(f"{Fore.LIGHTBLUE_EX} ** ExampleLoggingPlugin: on_epoch_end{Style.RESET_ALL}")
        for k, v in kwargs.items():
            logging.info(f"  {Fore.BLUE}{k}: {v}{Style.RESET_ALL}")
