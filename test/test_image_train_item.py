import unittest
import os
import pathlib
import PIL.Image as Image

from data.image_train_item import ImageCaption, ImageTrainItem

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
    
    def test_parse(self):
        caption = ImageCaption.parse("hello world, one,   two, three")

        self.assertEqual(caption.get_caption(), "hello world, one, two, three")
        
    def test_from_file_name(self):
        caption = ImageCaption.from_file_name("foo bar_1_2_3.jpg")
        self.assertEqual(caption.get_caption(), "foo bar")
        
    def test_from_text_file(self):
        caption = ImageCaption.from_text_file("test/data/test1.txt")
        self.assertEqual(caption.get_caption(), "caption for test1")
        
    def test_from_file(self):
        caption = ImageCaption.from_file("test/data/test1.txt")
        self.assertEqual(caption.get_caption(), "caption for test1")
        
        caption = ImageCaption.from_file("test/data/test_caption.caption")
        self.assertEqual(caption.get_caption(), "caption for test2")
        
    def test_resolve(self):
        caption = ImageCaption.resolve("test/data/test1.txt")
        self.assertEqual(caption.get_caption(), "caption for test1")
        
        caption = ImageCaption.resolve("test/data/test_caption.caption")
        self.assertEqual(caption.get_caption(), "caption for test2")
        
        caption = ImageCaption.resolve("hello world")
        self.assertEqual(caption.get_caption(), "hello world")
        
        caption = ImageCaption.resolve("test/data/test1.jpg")
        self.assertEqual(caption.get_caption(), "caption for test1")
        
        caption = ImageCaption.resolve("test/data/test2.jpg")
        self.assertEqual(caption.get_caption(), "test2")