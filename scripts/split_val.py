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

##  Script to move random files from source_root to destination_root for use as validation split
##  chooses (num_pairs_to_move) image/caption pairs from each subfolder in source_root and moves them to destination_root, preserving subfolder structure
##  also creates a validation_captions.txt file in destination_root for inference testing later to see how model performs on unseen data
##  only works on (png|jpg) + txt pairs.  Will break if there is no .txt file or images are other extensions

import os
import shutil
import random

source_root = "/mnt/nvme/mydata_train"
destination_root = "/mnt/nvme/mydata_val"

num_pairs_to_move = 3  # TODO: also do % based moving instead of fixed number

def move_random_file_pairs(source_folder, destination_folder):
    with open(os.path.join(destination_folder, "validation_captions.txt"), "w") as f:
        for subdir, dirs, files in os.walk(source_folder):
            for dir in dirs:
                source_subfolder = os.path.join(subdir, dir)
                destination_subfolder = os.path.join(destination_folder, dir)

                if not os.path.exists(destination_subfolder):
                    os.makedirs(destination_subfolder)

                file_list = [f for f in os.listdir(source_subfolder) if f.endswith((".png")) or f.endswith((".jpg"))]

                if len(file_list) >= num_pairs_to_move:
                    random.shuffle(file_list)

                    for i in range(num_pairs_to_move):
                        file_name = file_list[i]
                        source_file = os.path.join(source_subfolder, file_name)
                        destination_file = os.path.join(destination_subfolder, file_name)

                        caption_file_name = os.path.splitext(file_name)[0] + ".txt" 
                        caption_source_file = os.path.join(source_subfolder, caption_file_name)
                        caption_destination_file = os.path.join(destination_subfolder, caption_file_name)                 
                        
                        with open(caption_source_file, "r") as caption_source:
                            caption = caption_source.readline()
                            f.write(caption)
                            f.write("\n")
                        print(f"Moving {caption_source_file} to {caption_destination_file}")
                        print(f"Moving {source_file} to {destination_file}")
                        print(f"Caption: {caption}\n")
                        shutil.move(source_file, destination_file)
                        shutil.move(caption_source_file, caption_destination_file)


move_random_file_pairs(source_root, destination_root)
