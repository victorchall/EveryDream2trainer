from typing import Any, Dict
import torch
import bitsandbytes as bnb
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer, get_module_from_name
from transformers.modeling_utils import PreTrainedModel

# CogVLM stores inv_freq in the state dictionary but it is not in models._parameters so it cannot be quantized
# was patched in transformers for other models here: https://github.com/huggingface/transformers/pull/28837/files but cog is not part of transformers
def _patched_check_quantized_param(
        self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str, state_dict: Dict[str, Any], **kwargs
    ) -> bool:

    # if "inv_freq" in param_name:  # detect failure case
    #     print("check_quantized_param", param_name)

    module, tensor_name = get_module_from_name(model, param_name)
    if ("inv_freq" == tensor_name): # the fix
        return False
    if isinstance(module._parameters[tensor_name], bnb.nn.Params4bit): # will throw key error for inv_freq
        return True
    elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
        return True
    else:
        return False

def patch_cog():
    Bnb4BitHfQuantizer.check_quantized_param = _patched_check_quantized_param
