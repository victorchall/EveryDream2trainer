import unittest
import os
import pathlib
import PIL.Image as Image

from data.image_train_item import ImageCaption, ImageTrainItem
import data.aspects as aspects

DATA_PATH = pathlib.Path('./test/data')

class TestImageCaption(unittest.TestCase):
    
    def setUp(self) -> None:
        with open(DATA_PATH / "test1.txt", encoding='utf-8', mode='w') as f:
            f.write("caption for test1")
            
        Image.new("RGB", (512,512)).save(DATA_PATH / "test1.jpg")
        Image.new("RGB", (512,512)).save(DATA_PATH / "test2.jpg")
        
        with open(DATA_PATH / "test_caption.caption", encoding='utf-8', mode='w') as f:
            f.write("caption for test2")
            
        return super().setUp()
    
    def tearDown(self) -> None:
        for file in DATA_PATH.glob("test*"):
            file.unlink()

        return super().tearDown()
    
    def test_constructor(self):
        caption = ImageCaption("hello world", 1.0, ["one", "two", "three"], [1.0]*3, 2048, False)
        self.assertEqual(caption.get_caption(), "hello world, one, two, three")
        
        caption = ImageCaption("hello world", 1.0, [], [], 2048, False)
        self.assertEqual(caption.get_caption(), "hello world")

class TestImageTrainItemConstructor(unittest.TestCase):
    
    def tearDown(self) -> None:
        for file in DATA_PATH.glob("img_*"):
            file.unlink()

        return super().tearDown()

    @staticmethod 
    def image_with_size(width, height):
        filename = DATA_PATH / "img_{}x{}.jpg".format(width, height)
        Image.new("RGB", (width, height)).save(filename)
        caption = ImageCaption("hello world", 1.0, [], [], 2048, False)
        return ImageTrainItem(None, caption, aspects.ASPECTS_512, filename, 0.0, 1.0, False, False, 0)
           
    def test_target_size_computation(self):
        # Square images
        image = self.image_with_size(30, 30)
        self.assertEqual(image.target_wh, [512,512])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (30,30))

        image = self.image_with_size(512, 512)
        self.assertEqual(image.target_wh, [512,512])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (512,512))

        image = self.image_with_size(580, 580)
        self.assertEqual(image.target_wh, [512,512])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (580,580))

        # Landscape images
        image = self.image_with_size(64, 38)
        self.assertEqual(image.target_wh, [640,384])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (64,38))

        image = self.image_with_size(640, 384)
        self.assertEqual(image.target_wh, [640,384])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (640,384))

        image = self.image_with_size(704, 422)
        self.assertEqual(image.target_wh, [640,384])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (704,422))

        # Portrait images
        image = self.image_with_size(38, 64)
        self.assertEqual(image.target_wh, [384,640])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (38,64))

        image = self.image_with_size(384, 640)
        self.assertEqual(image.target_wh, [384,640])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (384,640))

        image = self.image_with_size(422, 704)
        self.assertEqual(image.target_wh, [384,640])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (422,704))

        
