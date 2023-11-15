import logging
import torch
from colorama import Fore, Style

def sigterm_handler(signum, frame):
    """
    handles sigterm
    """
    is_main_thread = (torch.utils.data.get_worker_info() == None)
    if is_main_thread:
        global interrupted
        if not interrupted:
            interrupted=True
            global global_step
            #TODO: save model on ctrl-c
            interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step}")
            print()
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
            logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
            time.sleep(2) # give opportunity to ctrl-C again to cancel save
            __save_model(interrupted_checkpoint_path, tokenizer, noise_scheduler, vae,
                            ed_optimizer, args.save_ckpt_dir, args.save_full_precision, args.save_optimizer,
                            save_ckpt=not args.no_save_ckpt)
        exit(_SIGTERM_EXIT_CODE)
    else:
        # non-main threads (i.e. dataloader workers) should exit cleanly
        exit(0)