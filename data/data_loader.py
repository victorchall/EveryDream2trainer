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
import bisect
import math
import copy

import random
from data.image_train_item import ImageTrainItem
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 715827880*4 # increase decompression bomb error limit to 4x default

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    data_root: root folder of training data
    batch_size: number of images per batch
    """
    def __init__(self, image_train_items, seed=555, batch_size=1):
        self.seed = seed
        self.batch_size = batch_size
        # Prepare data
        self.prepared_train_data = image_train_items
        random.Random(self.seed).shuffle(self.prepared_train_data)
        self.prepared_train_data = sorted(self.prepared_train_data, key=lambda img: img.caption.rating())
        # Initialize ratings
        self.rating_overall_sum: float = 0.0
        self.ratings_summed: list[float] = []
        for image in self.prepared_train_data:
            self.rating_overall_sum += image.caption.rating()
            self.ratings_summed.append(self.rating_overall_sum)

    def __pick_multiplied_set(self, randomizer):
        """
        Deals with multiply.txt whole and fractional numbers
        """
        #print(f"Picking multiplied set from {len(self.prepared_train_data)}")
        data_copy = copy.deepcopy(self.prepared_train_data) # deep copy to avoid modifying original multiplier property
        epoch_size = len(self.prepared_train_data)
        picked_images = []

        # add by whole number part first and decrement multiplier in copy
        for iti in data_copy:
            #print(f"check for whole number {iti.multiplier}: {iti.pathname}, remaining {iti.multiplier}")
            while iti.multiplier >= 1.0:
                picked_images.append(iti)
                #print(f"Adding {iti.multiplier}: {iti.pathname}, remaining {iti.multiplier}, , datalen: {len(picked_images)}")
                iti.multiplier -= 1.0

        remaining = epoch_size - len(picked_images)

        assert remaining >= 0, "Something went wrong with the multiplier calculation"

        # add by remaining fractional numbers by random chance
        while remaining > 0:
            for iti in data_copy:
                if randomizer.uniform(0.0, 1.0) < iti.multiplier:
                    #print(f"Adding {iti.multiplier}: {iti.pathname}, remaining {remaining}, datalen: {len(data_copy)}")
                    picked_images.append(iti)
                    remaining -= 1
                    iti.multiplier = 0.0
                if remaining <= 0:
                    break
        
        del data_copy
        return picked_images

    def get_shuffled_image_buckets(self, dropout_fraction: float = 1.0):
        """
        returns the current list of images including their captions in a randomized order,
        sorted into buckets with same sized images
        if dropout_fraction < 1.0, only a subset of the images will be returned
        if dropout_fraction >= 1.0, repicks fractional multipliers based on folder/multiply.txt values swept at prescan
        :param dropout_fraction: must be between 0.0 and 1.0.
        :return: randomized list of (image, caption) pairs, sorted into same sized buckets
        """

        self.seed += 1
        randomizer = random.Random(self.seed)

        if dropout_fraction < 1.0:
            picked_images = self.__pick_random_subset(dropout_fraction, randomizer)
        else:
            picked_images = self.__pick_multiplied_set(randomizer)

        randomizer.shuffle(picked_images)

        buckets = {}
        batch_size = self.batch_size
        for image_caption_pair in picked_images:
            image_caption_pair.runt_size = 0
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair)

        if len(buckets) > 1:
            for bucket in buckets:
                truncate_count = len(buckets[bucket]) % batch_size
                if truncate_count > 0:
                    runt_bucket = buckets[bucket][-truncate_count:]
                    for item in runt_bucket:
                        item.runt_size = truncate_count
                    while len(runt_bucket) < batch_size:
                        runt_bucket.append(random.choice(runt_bucket))

                    current_bucket_size = len(buckets[bucket])

                    buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                    buckets[bucket].extend(runt_bucket)

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    def __pick_random_subset(self, dropout_fraction: float, picker: random.Random) -> list[ImageTrainItem]:
        """
        Picks a random subset of all images
        - The size of the subset is limited by dropout_faction
        - The chance of an image to be picked is influenced by its rating. Double that rating -> double the chance
        :param dropout_fraction: must be between 0.0 and 1.0
        :param picker: seeded random picker
        :return: list of picked ImageTrainItem
        """

        prepared_train_data = self.prepared_train_data.copy()
        ratings_summed = self.ratings_summed.copy()
        rating_overall_sum = self.rating_overall_sum

        num_images = len(prepared_train_data)
        num_images_to_pick = math.ceil(num_images * dropout_fraction)
        num_images_to_pick = max(min(num_images_to_pick, num_images), 0)

        # logging.info(f"Picking {num_images_to_pick} images out of the {num_images} in the dataset for drop_fraction {dropout_fraction}")

        picked_images: list[ImageTrainItem] = []
        while num_images_to_pick > len(picked_images):
            # find random sample in dataset
            point = picker.uniform(0.0, rating_overall_sum)
            pos = min(bisect.bisect_left(ratings_summed, point), len(prepared_train_data) -1 )

            # pick random sample
            picked_image = prepared_train_data[pos]
            picked_images.append(picked_image)

            # kick picked item out of data set to not pick it again
            rating_overall_sum = max(rating_overall_sum - picked_image.caption.rating(), 0.0)
            ratings_summed.pop(pos)
            prepared_train_data.pop(pos)

        return picked_images
