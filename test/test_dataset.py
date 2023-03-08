import os
from data.dataset import Dataset, ImageConfig, Caption, Tag

from textwrap import dedent
from pyfakefs.fake_filesystem_unittest import TestCase

class TestResolve(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_simple_image(self):
        self.fs.create_file("image, tag1, tag2.jpg")

        actual = Dataset.from_path(".").image_configs

        expected = {
            ImageConfig(
                image="./image, tag1, tag2.jpg",
                captions=frozenset([
                    Caption(main_prompt="image", tags=frozenset([Tag("tag1"), Tag("tag2")]))
                ]))
        }
        self.assertEqual(expected, actual)

    def test_image_types(self):
        self.fs.create_file("image_1.JPG")
        self.fs.create_file("image_2.jpeg")
        self.fs.create_file("image_3.png")
        self.fs.create_file("image_4.webp")
        self.fs.create_file("image_5.jfif")
        self.fs.create_file("image_6.bmp")

        actual = Dataset.from_path(".").image_configs

        captions = frozenset([Caption(main_prompt="image")])
        expected = {
            ImageConfig(image="./image_1.JPG", captions=captions),
            ImageConfig(image="./image_2.jpeg", captions=captions),
            ImageConfig(image="./image_3.png", captions=captions),
            ImageConfig(image="./image_4.webp", captions=captions),
            ImageConfig(image="./image_5.jfif", captions=captions),
            ImageConfig(image="./image_6.bmp", captions=captions),
        }
        self.assertEqual(expected, actual)

    def test_caption_file(self):
        self.fs.create_file("image_1.jpg")
        self.fs.create_file("image_1.txt", contents="an image, test, from .txt")
        self.fs.create_file("image_2.jpg")
        self.fs.create_file("image_2.caption", contents="an image, test, from .caption")

        actual = Dataset.from_path(".").image_configs

        expected = {
            ImageConfig(
                image="./image_1.jpg",
                captions=frozenset([
                    Caption(main_prompt="an image", tags=frozenset([Tag("test"), Tag("from .txt")]))
                ])),
            ImageConfig(
                image="./image_2.jpg",
                captions=frozenset([
                    Caption(main_prompt="an image", tags=frozenset([Tag("test"), Tag("from .caption")]))
                ]))
        }
        self.assertEqual(expected, actual)


    def test_image_yaml(self):
        self.fs.create_file("image_1.jpg")
        self.fs.create_file("image_1.yaml", 
            contents=dedent("""
                multiply: 2
                cond_dropout: 0.05
                flip_p: 0.5
                caption: "A simple caption, from .yaml"
                """))
        self.fs.create_file("image_2.jpg")
        self.fs.create_file("image_2.yml", 
            contents=dedent("""
                flip_p: 0.0
                caption: 
                    main_prompt: A complex caption
                    rating: 1.1
                    max_caption_length: 1024
                    tags: 
                      - tag: from .yml
                      - tag: with weight
                        weight: 0.5
                """))

        actual = Dataset.from_path(".").image_configs

        expected = {
            ImageConfig(
                image="./image_1.jpg",
                multiply=2,
                cond_dropout=0.05,
                flip_p=0.5,
                captions=frozenset([
                    Caption(main_prompt="A simple caption", tags=frozenset([Tag("from .yaml")]))
                    ])),
            ImageConfig(
                image="./image_2.jpg",
                flip_p=0.0,
                captions=frozenset([
                    Caption(main_prompt="A complex caption", rating=1.1,
                    max_caption_length=1024,
                    tags=frozenset([
                        Tag("from .yml"), 
                        Tag("with weight", weight=0.5)
                        ]))
                    ]))
        }
        self.assertEqual(expected, actual)


    def test_multi_caption(self):
        self.fs.create_file("image_1.jpg")
        self.fs.create_file("image_1.yaml", contents=dedent("""
                caption: "A simple caption, from .yaml"
                captions: 
                    - "Another simple caption"
                    - main_prompt: A complex caption
                """))
        self.fs.create_file("image_1.txt", contents="A .txt caption")
        self.fs.create_file("image_1.caption", contents="A .caption caption")

        actual = Dataset.from_path(".").image_configs

        expected = {
            ImageConfig(
                image="./image_1.jpg",
                captions=frozenset([
                    Caption(main_prompt="A simple caption", tags=frozenset([Tag("from .yaml")])),
                    Caption(main_prompt="Another simple caption", tags=frozenset()),
                    Caption(main_prompt="A complex caption", tags=frozenset()),
                    Caption(main_prompt="A .txt caption", tags=frozenset()),
                    Caption(main_prompt="A .caption caption", tags=frozenset())
                    ])
                ),
        }
        self.assertEqual(expected, actual)

    def test_globals_and_locals(self):
        self.fs.create_file("./people/global.yaml", contents=dedent("""\
            multiply: 1.0
            cond_dropout: 0.0
            flip_p: 0.0
            """))
        self.fs.create_file("./people/alice/local.yaml", contents="multiply: 1.5")
        self.fs.create_file("./people/alice/alice_1.png")
        self.fs.create_file("./people/alice/alice_1.yaml", contents="multiply: 2")
        self.fs.create_file("./people/alice/alice_2.png")

        self.fs.create_file("./people/bob/multiply.txt", contents="3")
        self.fs.create_file("./people/bob/cond_dropout.txt", contents="0.05")
        self.fs.create_file("./people/bob/flip_p.txt", contents="0.05")
        self.fs.create_file("./people/bob/bob.png")

        self.fs.create_file("./people/cleo/cleo.png")
        self.fs.create_file("./people/dan.png")

        self.fs.create_file("./other/dog/local.yaml", contents="caption: spike")
        self.fs.create_file("./other/dog/xyz.png")

        actual = Dataset.from_path(".").image_configs

        expected = {
            ImageConfig(
                image="./people/alice/alice_1.png", 
                captions=frozenset([Caption(main_prompt="alice")]),
                multiply=2,
                cond_dropout=0.0,
                flip_p=0.0
                ),
            ImageConfig(
                image="./people/alice/alice_2.png", 
                captions=frozenset([Caption(main_prompt="alice")]),
                multiply=1.5,
                cond_dropout=0.0,
                flip_p=0.0
                ),
            ImageConfig(
                image="./people/bob/bob.png", 
                captions=frozenset([Caption(main_prompt="bob")]),
                multiply=3,
                cond_dropout=0.05,
                flip_p=0.05
                ),
            ImageConfig(
                image="./people/cleo/cleo.png", 
                captions=frozenset([Caption(main_prompt="cleo")]),
                multiply=1.0,
                cond_dropout=0.0,
                flip_p=0.0
                ),
            ImageConfig(
                image="./people/dan.png", 
                captions=frozenset([Caption(main_prompt="dan")]),
                multiply=1.0,
                cond_dropout=0.0,
                flip_p=0.0
                ),
            ImageConfig(
                image="./other/dog/xyz.png", 
                captions=frozenset([Caption(main_prompt="spike")]),
                multiply=None,
                cond_dropout=None,
                flip_p=None
                )
        }
        self.assertEqual(expected, actual) 

    def test_json_manifest(self):
        self.fs.create_file("./stuff/image_1.jpg")
        self.fs.create_file("./stuff/default.caption", contents= "default caption")
        self.fs.create_file("./other/image_1.jpg")
        self.fs.create_file("./other/image_2.jpg")
        self.fs.create_file("./other/image_3.jpg")
        self.fs.create_file("./manifest.json", contents=dedent("""
            [
                { "image": "./stuff/image_1.jpg", "caption": "./stuff/default.caption" },
                { "image": "./other/image_1.jpg", "caption": "other caption" },
                {
                    "image": "./other/image_2.jpg",
                    "caption": {
                        "main_prompt": "complex caption",
                        "rating": 0.1,
                        "max_caption_length": 1000,
                        "tags": [
                            {"tag": "including"},
                            {"tag": "weighted tag", "weight": 999.9}
                        ]
                    }
                },
                {
                    "image": "./other/image_3.jpg",
                    "multiply": 2,
                    "flip_p": 0.5,
                    "cond_dropout": 0.01,
                    "captions": [
                        "first caption",
                        { "main_prompt": "second caption" }
                    ]
                }
            ]
            """))

        actual = Dataset.from_json("./manifest.json").image_configs
        expected = {
            ImageConfig(
                image="./stuff/image_1.jpg",
                captions=frozenset([Caption(main_prompt="default caption")])
                ),
            ImageConfig(
                image="./other/image_1.jpg",
                captions=frozenset([Caption(main_prompt="other caption")])
                ),
            ImageConfig(
                image="./other/image_2.jpg",
                captions=frozenset([
                    Caption(
                        main_prompt="complex caption",
                        rating=0.1,
                        max_caption_length=1000,
                        tags=frozenset([
                            Tag("including"),
                            Tag("weighted tag", 999.9)
                        ]))
                    ])
                ),
            ImageConfig(
                image="./other/image_3.jpg",
                multiply=2,
                flip_p=0.5,
                cond_dropout=0.01,
                captions=frozenset([
                    Caption("first caption"),
                    Caption("second caption")
                    ])
                )
        }
        self.assertEqual(expected, actual)

    def test_train_items(self):
        dataset = Dataset([
            ImageConfig(
                image="1.jpg", 
                multiply=2,
                flip_p=0.1,
                cond_dropout=0.01,
                captions=frozenset([
                    Caption(
                        main_prompt="first caption",
                        rating = 1.1,
                        max_caption_length=1024,
                        tags=frozenset([
                            Tag("tag"),
                            Tag("tag_2", 2.0)
                        ])),
                    Caption(main_prompt="second_caption")
                ])),
            ImageConfig(
                image="2.jpg", 
                captions=frozenset([Caption(main_prompt="single caption")])
                )
        ])

        aspects = []
        actual = dataset.image_train_items(aspects)

        self.assertEqual(len(actual), 2)

        self.assertEqual(actual[0].pathname, os.path.abspath('1.jpg'))
        self.assertEqual(actual[0].multiplier, 2.0)
        self.assertEqual(actual[0].flip.p, 0.1) 
        self.assertEqual(actual[0].cond_dropout, 0.01)
        self.assertEqual(actual[0].caption.get_caption(), "first caption, tag, tag_2")
        # Can't test this
        # self.assertTrue(actual[0].caption.__use_weights)

        self.assertEqual(actual[1].pathname, os.path.abspath('2.jpg'))
        self.assertEqual(actual[1].multiplier, 1.0)
        self.assertEqual(actual[1].flip.p, 0.0)
        self.assertIsNone(actual[1].cond_dropout)
        self.assertEqual(actual[1].caption.get_caption(), "single caption")
        # Can't test this
        # self.assertFalse(actual[1].caption.__use_weights)