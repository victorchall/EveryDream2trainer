"""
Copyright [2022] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
import logging

def get_attn_yaml(ckpt_path):
    """
    Patch the UNet to use updated attention heads for xformers support in FP32
    """
    unet_cfg_path = os.path.join(ckpt_path, "unet", "config.json")
    with open(unet_cfg_path, "r") as f:
        unet_cfg = json.load(f)

    scheduler_cfg_path = os.path.join(ckpt_path, "scheduler", "scheduler_config.json")
    with open(scheduler_cfg_path, "r") as f:
        scheduler_cfg = json.load(f)

    is_sd1attn = unet_cfg["attention_head_dim"] == [8, 8, 8, 8]
    is_sd1attn = unet_cfg["attention_head_dim"] == 8 or is_sd1attn

    prediction_type = scheduler_cfg["prediction_type"]

    logging.info(f" unet attention_head_dim: {unet_cfg['attention_head_dim']}")

    yaml = ''
    if prediction_type in ["v_prediction","v-prediction"] and not is_sd1attn:
        yaml = "v2-inference-v.yaml"
    elif prediction_type == "epsilon" and not is_sd1attn:
        yaml = "v2-inference.yaml"
    elif prediction_type == "epsilon" and is_sd1attn:
        yaml = "v1-inference.yaml"
    else:
        raise ValueError(f"Unknown model format for: {prediction_type} and attention_head_dim {unet_cfg['attention_head_dim']}")

    logging.info(f"Inferred yaml: {yaml}, attn: {'sd1' if is_sd1attn else 'sd2'}, prediction_type: {prediction_type}")

    return is_sd1attn, yaml
