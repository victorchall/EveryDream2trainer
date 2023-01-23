import json
import glob
import os
import unittest

import PIL.Image as Image

import data.aspects as aspects
import data.resolver as resolver

DATA_PATH = os.path.abspath('./test/data')
JSON_ROOT_PATH = os.path.join(DATA_PATH, 'test_root.json')
ASPECTS = aspects.get_aspect_buckets(512)
FLIP_P = 0.0

IMAGE_1_PATH = os.path.join(DATA_PATH, 'test1.jpg')
CAPTION_1_PATH = os.path.join(DATA_PATH, 'test1.txt')
IMAGE_2_PATH = os.path.join(DATA_PATH, 'test2.jpg')
IMAGE_3_PATH = os.path.join(DATA_PATH, 'test3.jpg')

class TestResolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Image.new('RGB', (512, 512)).save(IMAGE_1_PATH)
        with open(CAPTION_1_PATH, 'w') as f:
            f.write('caption for test1')

        Image.new('RGB', (512, 512)).save(IMAGE_2_PATH)
        # Undersized image. Should cause an event.
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

    def setUp(self) -> None:
        self.events = []
        self.on_event = lambda event: self.events.append(event.name)
        return super().setUp()   
    
    def tearDown(self) -> None:
        self.events = []
        self.on_event = None
        return super().tearDown()
        
    def test_directory_resolve_with_str(self):
        image_train_items = resolver.resolve(DATA_PATH, ASPECTS, FLIP_P, self.on_event)
        image_paths = [item.pathname for item in image_train_items]
        image_captions = [item.caption for item in image_train_items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(image_train_items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'test2', 'test3'])
        self.assertEqual(self.events, ['undersized_image'])
    
    def test_directory_resolve_with_dict(self):
        data_root_spec = {
            'resolver': 'directory',
            'path': DATA_PATH,
        }
        
        image_train_items = resolver.resolve(data_root_spec, ASPECTS, FLIP_P, self.on_event)
        image_paths = [item.pathname for item in image_train_items]
        image_captions = [item.caption for item in image_train_items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(image_train_items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'test2', 'test3'])
        self.assertEqual(self.events, ['undersized_image'])
    
    def test_json_resolve_with_str(self):
        image_train_items = resolver.resolve(JSON_ROOT_PATH, ASPECTS, FLIP_P, self.on_event)
        image_paths = [item.pathname for item in image_train_items]
        image_captions = [item.caption for item in image_train_items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(image_train_items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'caption for test2', 'test3'])
        self.assertEqual(self.events, ['undersized_image'])
    
    def test_json_resolve_with_dict(self):
        data_root_spec = {
            'resolver': 'json',
            'path': JSON_ROOT_PATH,
        }
        
        image_train_items = resolver.resolve(data_root_spec, ASPECTS, FLIP_P, self.on_event)
        image_paths = [item.pathname for item in image_train_items]
        image_captions = [item.caption for item in image_train_items]
        captions = [caption.get_caption() for caption in image_captions]
        
        self.assertEqual(len(image_train_items), 3)
        self.assertEqual(image_paths, [IMAGE_1_PATH, IMAGE_2_PATH, IMAGE_3_PATH])
        self.assertEqual(captions, ['caption for test1', 'caption for test2', 'test3'])
        self.assertEqual(self.events, ['undersized_image'])