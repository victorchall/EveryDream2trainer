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
from PIL import Image
import random
from data.image_train_item import ImageTrainItem
import data.aspects as aspects

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(self, data_root, seed=555, debug_level=0, batch_size=1, flip_p=0.0, resolution=512):
        self.image_paths = []
        self.debug_level = debug_level
        self.flip_p = flip_p

        self.aspects = aspects.get_aspect_buckets(resolution=resolution, square_only=False)
        print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        print(" Preloading images...")

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        prepared_train_data = self.__prescan_images(self.image_paths, flip_p) # ImageTrainItem[]
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)

        #if debug_level > 0: print(f" * DLMA Example: {self.image_caption_pairs[0]} images")

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __read_caption_from_file(file_path, fallback_caption):
        caption = fallback_caption
        try:
            with open(file_path, encoding='utf-8', mode='r') as caption_file:
                caption = caption_file.read()
        except:
            print(f" *** Error reading {file_path} to get caption, falling back to filename")
            caption = fallback_caption
            pass
        return caption

    def __prescan_images(self, image_paths: list, flip_p=0.0):
        """
        Create ImageTrainItem objects with metadata for hydration later
        """
        decorated_image_train_items = []

        for pathname in image_paths:
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]

            txt_file_path = os.path.splitext(pathname)[0] + ".txt"
            caption_file_path = os.path.splitext(pathname)[0] + ".caption"

            if os.path.exists(txt_file_path):
                caption = self.__read_caption_from_file(txt_file_path, caption_from_filename)                
            elif os.path.exists(caption_file_path):
                caption = self.__read_caption_from_file(caption_file_path, caption_from_filename)                
            else:
                caption = caption_from_filename

            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=caption, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            decorated_image_train_items.append(image_train_item)

        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair) 

        print(f" ** Number of buckets used: {len(buckets)}")

        if len(buckets) > 1:
            for bucket in buckets:
                truncate_count = len(buckets[bucket]) % batch_size
                current_bucket_size = len(buckets[bucket])
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]

                if debug_level > 0:
                    print(f"  ** Bucket {bucket} with {current_bucket_size} will drop {truncate_count} images due to batch size {batch_size}")

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root):
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1]
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.image_paths.append(current)

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)
