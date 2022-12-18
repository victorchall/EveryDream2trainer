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
import torch
import os
import hashlib
import io
from PIL import Image, ImageOps
import random
from aspects import get_aspect_buckets
from torchvision import transforms

class LatentCacheItem():
    """
    caches image/caption latent pairs and index value to select appropriate random crop jitter
    """
    def __init__(self, imagelatent, captionembedding, cropjitteridx, resolution = tuple):
        """
        imagelatent: image tensor
        captionembedding: caption embedding tensor
        cropjitteridx: index of random crop jitter to use
        """
        self.imagelatent = imagelatent
        self.captionembedding = captionembedding
        self.cropjitteridx = cropjitteridx
        self.resolution = resolution
        
    def __repr__(self):
        return f"lat: {self.imagelatent.shape} emb:{self.captionembedding.shape} cj:{self.cropjitteridx}"

class LatentCacheManager():
    """
    Manages a cache of latent vectors for a dataset.
    """
    def __init__(self, latent_cache_path="/.cache/latents", device=torch.device("cuda"), jitter_lim=8, vae=None):
        """
        Manages caching of image latents to disk,
        latent_cache_path: path to latent cache folder
        device: device to use for creating latents (torch.device)
        vae: vae to use for creating latents
        jitter_lim: number of random crop jitters to use per image (default: 8)
        """
        assert vae is not None, "LatentCacheManager requires a vae to be passed in"

        self.cache = dict(str, []) # key: sha256 hash of image path, value: list of LatentCacheItem
        self.latentcachepath = latent_cache_path
        self.jitter_lim = jitter_lim
        self.device = device
        self.vae = vae

        # create pt file if it doesn't exist
        if not os.path.exists(self.latentcachepath):
            torch.save(self.cache, self.latentcachepath)
        
        self.vae_on_device = False

    def set_vae(self, vae):
        self.vae = vae
    
    def delete_vae(self):
        self.vae = None

    def vae_to_device(self, device):
        self.vae.to(self.device)
        self.vae_on_device = True
    
    def vae_to_cpu(self):
        self.vae.to("cpu")
        self.vae_on_device = False

    @staticmethod
    def __hash(imagepath):
        return hashlib.sha256(imagepath.encode("utf-8")).hexdigest()
    
    def add(self, imagepath: io, captionembedding: torch.tensor, target_resolution=(512,512)):
        """
        adds aan item to the cache
        """
        if not self.vae_on_device: self.vae_to_gpu()
        hash = self.__hash(imagepath)

        image = Image.open(imagepath)
        image_aspects = get_aspect_buckets(resolution=target_resolution)

        for i in range(self.jitter_lim):
            bleed = random.uniform(0.0, 0.02)
            centering = (random.uniform(0.0, 0.02), random.uniform(0.0, 0.02))
            jittered_image = ImageOps.fit(image, target_resolution, method=Image.BICUBIC, bleed=bleed, centering=centering)
            # convert to tensor
            latent = self.vae(jittered_image)
            # add to cache
            self.cache[hash].append(LatentCacheItem(imagelatent=latent, 
                                                    captionembedding=captionembedding,
                                                    i, 
                                                    resolution=self.vae.resolution))


        
        # append to pt file 
        torch.save(self.cache, os.path.join(self.latentcachepath, f"{hash}.pt"))

    def __getitem__(self, imagepath, cropjitteridx=0):
        """
        returns a LatentCacheItem by imagepath key
        """
        hash = self.__hash(imagepath)

        item = self.cache[hash][cropjitteridx]
        return item
