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

class LogWrapper(object):
    """
    singleton for logging
    """
    def __init__(self, log_dir, project_name):
        self.log_dir = log_dir

        start_time = time.strftime("%Y%m%d-%H%M")
        self.log_file = os.path.join(log_dir, f"log-{project_name}-{start_time}.txt")

        self.logger = logging.getLogger(__name__)

        console = logging.StreamHandler()
        self.logger.addHandler(console)

        file = logging.FileHandler(self.log_file, mode="a", encoding=None, delay=False)
        self.logger.addHandler(file)

    def __call__(self):
        return self.logger
