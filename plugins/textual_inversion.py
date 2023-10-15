import json
import logging
import os.path

import torch
from colorama import Fore

from plugins.plugins import BasePlugin
from train import EveryDreamTrainingState

""" config file format:

{
  'tokens': [
    { 'token': 'hat*', 'initializer': 'hat' },
    { 'token': 'dancing shoes', 'initializer_word': 'shoes' }, # multi word string ok
  ]
}
"""

class TextualInversionPlugin(BasePlugin):

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "textual_inversion.json")
        print(f"Textual Inversion plugin instantiated, loading data from {path}...")
        with open(path, 'rt') as f:
            self.config = json.load(f)

    def on_model_load(self, **kwargs):
        ed_state: EveryDreamTrainingState = kwargs.get('ed_state')
        optimizer_config: dict = kwargs.get('optimizer_config')

        # check for correctly configured text encoder training
        num_te_layers = len(ed_state.text_encoder.text_model.encoder.layers)
        if (optimizer_config is None or
            'text_encoder_freezing' not in optimizer_config or
            optimizer_config['text_encoder_freezing'].get('freeze_embeddings') != False or
            optimizer_config['text_encoder_freezing'].get('unfreeze_last_n_layers', 0) < num_te_layers
        ):
            required_js_fragment = {"text_encoder_freezing": {"freeze_embeddings": False, "unfreeze_last_n_layers": num_te_layers}}
            logging.error(f" * {Fore.LIGHTRED_EX}Textual Inversion plugin REQUIRES the following json fragment in your optimizer config:{Fore.RESET}")
            logging.error(f" * {Fore.LIGHTRED_EX}  {json.dumps(required_js_fragment)}{Fore.RESET}")
            raise RuntimeError("Misconfigured optimizer config")

        tokens_to_add = [t['token'] for t in self.config['tokens']]
        num_added_tokens = ed_state.tokenizer.add_tokens(tokens_to_add)
        if num_added_tokens != len(tokens_to_add):
            raise RuntimeError(f"Tokens not added successfully - tried to add {len(tokens_to_add)} but only added {num_added_tokens}")
        ed_state.text_encoder.resize_token_embeddings(len(ed_state.tokenizer))

        added_token_ids = []
        input_embeddings = ed_state.text_encoder.get_input_embeddings()
        for token_info in self.config['tokens']:
            # get newly added token id
            t = token_info['token']
            token_ids = ed_state.tokenizer.convert_tokens_to_ids(ed_state.tokenizer.tokenize(t))
            if len(token_ids) != 1:
                raise RuntimeError(f"Tokens not added succesfully - expected 1 token id for {t}, found {len(token_ids)}")
            token_id = token_ids[0]
            added_token_ids.append(token_id)

            # copy initializer embedding
            initializer_word = token_info['initializer_word']
            initializer_word_token_ids = ed_state.tokenizer.convert_tokens_to_ids(ed_state.tokenizer.tokenize(initializer_word))
            if len(initializer_word_token_ids) != 1:
                raise RuntimeError(f"Tokens not added succesfully - initializer word '{initializer_word}' needs "
                                   f"{len(initializer_word_token_ids)} tokens, but only single tokens are supported.")
            initializer_word_token_id = initializer_word_token_ids[0]
            initializer_embedding = input_embeddings.weight.data[initializer_word_token_id]
            input_embeddings.weight.data[token_id] = initializer_embedding

        self.added_token_ids = added_token_ids
        self.original_text_embeddings = ed_state.text_encoder.get_input_embeddings().weight.data.detach().clone()


    def on_step_end(self, **kwargs):
        ed_state: EveryDreamTrainingState = kwargs['ed_state']
        # reset all embeddings except the ones we're training to their original state
        index_no_updates = torch.isin(torch.arange(len(ed_state.tokenizer)), torch.Tensor(self.added_token_ids))
        with (torch.no_grad()):
            ed_state.text_encoder.get_input_embeddings().weight[
                index_no_updates
            ] = self.original_text_embeddings[index_no_updates]
