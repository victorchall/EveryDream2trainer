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
import logging
import os
import time
from colorama import Fore, Style

from tensorboard import SummaryWriter
import wandb

class LogWrapper():
    """
    singleton for logging
    """
    def __init__(self, args, wandb=False):
        self.logdir = args.logdir
        self.wandb = wandb

        if wandb:
            wandb.init(project=args.project_name, sync_tensorboard=True)
        else:
            self.log_writer = SummaryWriter(log_dir=args.logdir, 
                                    flush_secs=5,
                                    comment="EveryDream2FineTunes",
                                    )

        start_time = time.strftime("%Y%m%d-%H%M")
        log_file = os.path.join(args.logdir, f"log-{args.project_name}-{start_time}.txt")

        self.logger = logging.getLogger(__name__)

        console = logging.StreamHandler()
        self.logger.addHandler(console)

        file = logging.FileHandler(log_file, mode="a", encoding=None, delay=False)
        self.logger.addHandler(file)

    def add_image():
        """
        log_writer.add_image(tag=f"sample_{i}", img_tensor=tfimage, global_step=gs)
            else:
                log_writer.add_image(tag=f"sample_{i}_{clean_prompt[:100]}", img_tensor=tfimage, global_step=gs)
        """
        pass

    def add_scalar(self, tag: str, img_tensor: float, global_step: int):
        if self.wandb:
            wandb.log({tag: img_tensor}, step=global_step)
        else:
            self.log_writer.add_image(tag, img_tensor, global_step)

    def append_epoch_log(self, global_step: int, epoch_pbar, gpu, log_writer, **logs):
        """
        updates the vram usage for the epoch
        """
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        self.add_scalar("performance/vram", gpu_used_mem, global_step)
        epoch_mem_color = Style.RESET_ALL
        if gpu_used_mem > 0.93 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTRED_EX
        elif gpu_used_mem > 0.85 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTYELLOW_EX
        elif gpu_used_mem > 0.7 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTGREEN_EX
        elif gpu_used_mem < 0.5 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTBLUE_EX

        if logs is not None:
            epoch_pbar.set_postfix(**logs, vram=f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step}")