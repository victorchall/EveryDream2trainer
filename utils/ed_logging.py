import logging
import argparse

def configure_logging(args: argparse.Namespace, log_file=None):
    level = logging.INFO if not args.debug else logging.DEBUG

    if log_file:
        filemode = "a" if args.append_log else "w"
        logging.basicConfig(filename=log_file, 
                            level=level, 
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            filemode=filemode)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)