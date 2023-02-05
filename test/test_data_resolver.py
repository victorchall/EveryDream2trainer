import json
import glob
import os
import unittest
import argparse

import PIL.Image as Image

import data.aspects as aspects
import data.resolver as resolver

DATA_PATH = os.path.abspath('./test/data')
JSON_ROOT_PATH = os.path.join(DATA_PATH, 'test_root.json')

IMAGE_1_PATH = os.path.join(DATA_PATH, 'test1.jpg')
CAPTION_1_PATH = os.path.join(DATA_PATH, 'test1.txt')
IMAGE_2_PATH = os.path.join(DATA_PATH, 'test2.jpg')
IMAGE_3_PATH = os.path.join(DATA_PATH, 'test3.jpg')

ARGS = argparse.Namespace(
    aspects=aspects.get_aspect_buckets(512),
    flip_p=0.5,
    seed=42,
)

class TestResolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Image.new('RGB', (512, 512)).save(IMAGE_1_PATH)
        with open(CAPTION_1_PATH, 'w') as f:
            f.write('caption for test1')

        Image.new('RGB', (512, 512)).save(IMAGE_2_PATH)
        # Undersized image
        Image.new('RGB', (256, 256)).save(IMAGE_3_PATH)
        
        json_data = [
            {
                'image': IMAGE_1_PATH,
                'caption': CAPTION_1_PATH
            },
            {
                'image': IMAGE_2_PATH,
                'caption': 'caption for test2'
            },
            {
                'image': IMAGE_3_PATH,
            }
        ]
        
        with open(JSON_ROOT_PATH, 'w') as f:
            json.dump(json_data, f, indent=4)
            
    @classmethod
    def tearDownClass(cls):
        for file in glob.glob(os.path.join(DATA_PATH, 'test*')):
            os.remove(file)    

    def test_directory_resolve_with_str(self):
        items = resolver.resolve(DATA_PATH, ARGS)
        image_paths = [item.pathname for item in items]
        image_captions = [item.caption for item in items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'test2', 'test3'])
        
        undersized_images = list(filter(lambda i: i.is_undersized, items))
        self.assertEqual(len(undersized_images), 1)
    
    def test_directory_resolve_with_dict(self):
        data_root_spec = {
            'resolver': 'directory',
            'path': DATA_PATH,
        }
        
        items = resolver.resolve(data_root_spec, ARGS)
        image_paths = [item.pathname for item in items]
        image_captions = [item.caption for item in items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'test2', 'test3'])
        
        undersized_images = list(filter(lambda i: i.is_undersized, items))
        self.assertEqual(len(undersized_images), 1)
    
    def test_json_resolve_with_str(self):
        items = resolver.resolve(JSON_ROOT_PATH, ARGS)
        image_paths = [item.pathname for item in items]
        image_captions = [item.caption for item in items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'caption for test2', 'test3'])
        
        undersized_images = list(filter(lambda i: i.is_undersized, items))
        self.assertEqual(len(undersized_images), 1)
    
    def test_json_resolve_with_dict(self):
        data_root_spec = {
            'resolver': 'json',
            'path': JSON_ROOT_PATH,
        }
        
        items = resolver.resolve(data_root_spec, ARGS)
        image_paths = [item.pathname for item in items]
        image_captions = [item.caption for item in items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'caption for test2', 'test3'])
        
        undersized_images = list(filter(lambda i: i.is_undersized, items))
        self.assertEqual(len(undersized_images), 1)
        
    def test_resolve_with_list(self):
        data_root_spec = [
            DATA_PATH,
            JSON_ROOT_PATH,
        ]
        
        items = resolver.resolve(data_root_spec, ARGS)
        image_paths = [item.pathname for item in items]
        image_captions = [item.caption for item in items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(items), 6)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH] * 2)
        self.assertEqual(captions, ['caption for test1', 'test2', 'test3', 'caption for test1', 'caption for test2', 'test3'])
        
        undersized_images = list(filter(lambda i: i.is_undersized, items))
        self.assertEqual(len(undersized_images), 2)