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
import logging
import math
import copy

import random
from typing import List, Dict

from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID
import PIL.Image

from utils.first_fit_decreasing import first_fit_decreasing

PIL.Image.MAX_IMAGE_PIXELS = 715827880*4 # increase decompression bomb error limit to 4x default

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    image_train_items: list of `ImageTrainItem` objects
    seed: random seed
    batch_size: number of images per batch
    """
    def __init__(self, image_train_items: list[ImageTrainItem], seed=555, batch_size=1, grad_accum=1):
        self.seed = seed
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.prepared_train_data = image_train_items
        random.Random(self.seed).shuffle(self.prepared_train_data)
        self.prepared_train_data = sorted(self.prepared_train_data, key=lambda img: img.caption.rating())
        self.expected_epoch_size = math.floor(sum([i.multiplier for i in self.prepared_train_data]))
        if self.expected_epoch_size != len(self.prepared_train_data):
            logging.info(f" * DLMA initialized with {len(image_train_items)} source images. After applying multipliers, each epoch will train on at least {self.expected_epoch_size} images.")
        else:
            logging.info(f" * DLMA initialized with {len(image_train_items)} images.")

        self.rating_overall_sum: float = 0.0
        self.ratings_summed: list[float] = []
        self.__update_rating_sums()


    def __pick_multiplied_set(self, randomizer: random.Random):
        """
        Deals with multiply.txt whole and fractional numbers
        """
        picked_images = []
        data_copy = copy.deepcopy(self.prepared_train_data) # deep copy to avoid modifying original multiplier property
        for iti in data_copy:
            while iti.multiplier >= 1:
                picked_images.append(iti)
                iti.multiplier -= 1

        remaining = self.expected_epoch_size - len(picked_images)

        assert remaining >= 0, "Something went wrong with the multiplier calculation"

        # resolve fractional parts, ensure each is only added max once
        while remaining > 0:
            for iti in data_copy:
                if randomizer.random() < iti.multiplier:
                    picked_images.append(iti)
                    iti.multiplier = 0
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        return picked_images

    def get_shuffled_image_buckets(self, dropout_fraction: float = 1.0) -> list[ImageTrainItem]:
        """
        Returns the current list of `ImageTrainItem` in randomized order,
        sorted into buckets with same sized images.
        
        If dropout_fraction < 1.0, only a subset of the images will be returned.
        
        If dropout_fraction >= 1.0, repicks fractional multipliers based on folder/multiply.txt values swept at prescan.
        
        :param dropout_fraction: must be between 0.0 and 1.0.
        :return: Randomized list of `ImageTrainItem` objects
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
        grad_accum = self.grad_accum

        def add_image_to_appropriate_bucket(image: ImageTrainItem, batch_id_override: str=None):
            bucket_key = (image.batch_id if batch_id_override is None else batch_id_override,
                          image.target_wh[0],
                          image.target_wh[1])
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(image)

        for image_caption_pair in picked_images:
            image_caption_pair.runt_size = 0
            add_image_to_appropriate_bucket(image_caption_pair)

        # handled named batch runts by demoting them to the DEFAULT_BATCH_ID
        for key, bucket_contents in [(k, b) for k, b in buckets.items() if k[0] != DEFAULT_BATCH_ID]:
            runt_count = len(bucket_contents) % batch_size
            if runt_count == 0:
                continue
            runts = bucket_contents[-runt_count:]
            del bucket_contents[-runt_count:]
            for r in runts:
                add_image_to_appropriate_bucket(r, batch_id_override=DEFAULT_BATCH_ID)
            if len(bucket_contents) == 0:
                del buckets[key]

        # handle remaining runts by randomly duplicating items
        for bucket in buckets:
            truncate_count = len(buckets[bucket]) % batch_size
            if truncate_count > 0:
                assert bucket[0] == DEFAULT_BATCH_ID, "there should be no more runts in named batches"
                runt_bucket = buckets[bucket][-truncate_count:]
                for item in runt_bucket:
                    item.runt_size = truncate_count
                while len(runt_bucket) < batch_size:
                    runt_bucket.append(random.choice(runt_bucket))

                current_bucket_size = len(buckets[bucket])

                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                buckets[bucket].extend(runt_bucket)

        items_by_batch_id = collapse_buckets_by_batch_id(buckets)
        # at this point items have a partially deterministic order
        # (in particular: rarer aspect ratios are more likely to cluster at the end due to stochastic sampling)
        # so we shuffle them to mitigate this, using chunked_shuffle to keep batches with the same aspect ratio together
        items_by_batch_id = {k: chunked_shuffle(v, chunk_size=batch_size, randomizer=randomizer)
                             for k,v in items_by_batch_id.items()}
        # paranoia: verify that this hasn't fucked up the aspect ratio batching
        for items in items_by_batch_id.values():
            batches = chunk(items, chunk_size=batch_size)
            for batch in batches:
                target_wh = batch[0].target_wh
                assert all(target_wh == i.target_wh for i in batch[1:]), "mixed aspect ratios in a batch - this shouldn't happen"

        # handle batch_id
        # unlabelled data (no batch_id) is in batches labelled DEFAULT_BATCH_ID.
        items = flatten_buckets_preserving_named_batch_adjacency(items_by_batch_id,
                                                                   batch_size=batch_size,
                                                                   grad_accum=grad_accum)

        effective_batch_size = batch_size * grad_accum
        items = chunked_shuffle(items, chunk_size=effective_batch_size, randomizer=randomizer)

        return items


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

    def __update_rating_sums(self):        
        self.rating_overall_sum: float = 0.0
        self.ratings_summed: list[float] = []
        for item in self.prepared_train_data:
            self.rating_overall_sum += item.caption.rating()
            self.ratings_summed.append(self.rating_overall_sum)


def chunk(l: List, chunk_size) -> List:
    num_chunks = int(math.ceil(float(len(l)) / chunk_size))
    return [l[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

def unchunk(chunked_list: List):
    return [i for c in chunked_list for i in c]

def collapse_buckets_by_batch_id(buckets: Dict) -> Dict:
    batch_ids = [k[0] for k in buckets.keys()]
    items_by_batch_id = {}
    for batch_id in batch_ids:
        items_by_batch_id[batch_id] = unchunk([b for bucket_key,b in buckets.items() if bucket_key[0] == batch_id])
    return items_by_batch_id

def flatten_buckets_preserving_named_batch_adjacency(items_by_batch_id: Dict[str, List[ImageTrainItem]],
                                                       batch_size: int,
                                                       grad_accum: int) -> List[ImageTrainItem]:
    # precondition: items_by_batch_id has no incomplete batches
    assert(all((len(v) % batch_size)==0 for v in items_by_batch_id.values()))
    # ensure we don't mix up aspect ratios by treating each chunk of batch_size images as
    # a single unit to pass to first_fit_decreasing()
    filler_items = chunk(items_by_batch_id.get(DEFAULT_BATCH_ID, []), batch_size)
    custom_batched_items = [chunk(v, batch_size) for k, v in items_by_batch_id.items() if k != DEFAULT_BATCH_ID]
    neighbourly_chunked_items = first_fit_decreasing(custom_batched_items,
                                                     batch_size=grad_accum,
                                                     filler_items=filler_items)

    items: List[ImageTrainItem] = unchunk(neighbourly_chunked_items)
    return items

def chunked_shuffle(l: List, chunk_size: int, randomizer: random.Random) -> List:
    """
    Shuffles l in chunks, preserving the chunk boundaries and the order of items within each chunk.
    If the last chunk is incomplete, it is not shuffled (i.e. preserved as the last chunk)
    """
    if len(l) == 0:
        return []

    # chunk by effective batch size
    chunks = chunk(l, chunk_size)
    # preserve last chunk as last if it is incomplete
    last_chunk = None
    if len(chunks[-1]) < chunk_size:
        last_chunk = chunks.pop(-1)
    randomizer.shuffle(chunks)
    if last_chunk is not None:
        chunks.append(last_chunk)
    l = unchunk(chunks)
    return l
