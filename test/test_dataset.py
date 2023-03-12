import os
from data.dataset import Dataset, ImageConfig, Tag

from textwrap import dedent
from pyfakefs.fake_filesystem_unittest import TestCase

class TestDataset(TestCase):
    
    def setUp(self):
        self.maxDiff = None
        self.setUpPyfakefs()

    def test_a_caption_is_generated_from_image_given_no_other_config(self):
        self.fs.create_file("image, tag1, tag2.jpg")

        actual = Dataset.from_path(".").image_configs

        expected = {
            "./image, tag1, tag2.jpg": ImageConfig(main_prompts="image", tags=frozenset([Tag("tag1"), Tag("tag2")]))
        }
        self.assertEqual(expected, actual)

    def test_several_image_formats_are_supported(self):
        self.fs.create_file("image.JPG")
        self.fs.create_file("image.jpeg")
        self.fs.create_file("image.png")
        self.fs.create_file("image.webp")
        self.fs.create_file("image.jfif")
        self.fs.create_file("image.bmp")

        actual = Dataset.from_path(".").image_configs

        common_cfg = ImageConfig(main_prompts="image")
        expected = {
            "./image.JPG": common_cfg,
            "./image.jpeg": common_cfg,
            "./image.png": common_cfg,
            "./image.webp": common_cfg,
            "./image.jfif": common_cfg,
            "./image.bmp": common_cfg,
        }
        self.assertEqual(expected, actual)

    def test_captions_can_be_read_from_txt_or_caption_sidecar(self):
        self.fs.create_file("image_1.jpg")
        self.fs.create_file("image_1.txt", contents="an image, test, from .txt")
        self.fs.create_file("image_2.jpg")
        self.fs.create_file("image_2.caption", contents="an image, test, from .caption")

        actual = Dataset.from_path(".").image_configs

        expected = {
            "./image_1.jpg": ImageConfig(main_prompts="an image", tags=frozenset([Tag("test"), Tag("from .txt")])),
            "./image_2.jpg": ImageConfig(main_prompts="an image", tags=frozenset([Tag("test"), Tag("from .caption")]))
        }
        self.assertEqual(expected, actual)


    def test_captions_and_options_can_be_read_from_yaml_sidecar(self):
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
            "./image_1.jpg": ImageConfig(
                multiply=2,
                cond_dropout=0.05,
                flip_p=0.5,
                main_prompts="A simple caption", 
                tags= { Tag("from .yaml") }
            ),
            "./image_2.jpg": ImageConfig(
                flip_p=0.0,
                rating=1.1,
                max_caption_length=1024,
                main_prompts="A complex caption", 
                tags= { Tag("from .yml"), Tag("with weight", weight=0.5) }
                )
            }
        self.assertEqual(expected, actual)


    def test_multiple_prompts_and_tags_from_multiple_sidecars_are_supported(self):
        self.fs.create_file("image_1.jpg")
        self.fs.create_file("image_1.yaml", contents=dedent("""
                main_prompt: 
                    - unique prompt
                    - dupe prompt
                tags: 
                    - from .yaml
                    - dupe tag 
                """))
        self.fs.create_file("image_1.txt", contents="also unique prompt, from .txt, dupe tag")
        self.fs.create_file("image_1.caption", contents="dupe prompt, from .caption")

        actual = Dataset.from_path(".").image_configs

        expected = {
            "./image_1.jpg": ImageConfig( 
                main_prompts={ "unique prompt", "also unique prompt", "dupe prompt" },
                tags={ Tag("from .yaml"), Tag("from .txt"), Tag("from .caption"), Tag("dupe tag") }
                )
            }
        self.assertEqual(expected, actual)

    def test_sidecars_can_also_be_attached_to_local_and_recursive_folders(self):
        self.fs.create_file("./global.yaml", 
            contents=dedent("""\
                main_prompt: global prompt
                tags:
                    - global tag
                flip_p: 0.0
                """))

        self.fs.create_file("./local.yaml", 
            contents=dedent("""
                main_prompt: local prompt
                tags: 
                    - tag: local tag
                """))

        self.fs.create_file("./arbitrary filename.png")
        self.fs.create_file("./sub/sub arbitrary filename.png")
        self.fs.create_file("./sub/sidecar.png")
        self.fs.create_file("./sub/sidecar.txt", 
            contents="sidecar prompt, sidecar tag")

        self.fs.create_file("./optfile/optfile.png")
        self.fs.create_file("./optfile/flip_p.txt",
            contents="0.1234")

        self.fs.create_file("./sub/sub2/global.yaml", 
            contents=dedent("""
            tags: 
                - tag: sub global tag
            """))
        self.fs.create_file("./sub/sub2/local.yaml", 
            contents=dedent("""
                tags: 
                    - This tag wil not apply to any files
            """))
        self.fs.create_file("./sub/sub2/sub3/xyz.png")

        actual = Dataset.from_path(".").image_configs

        expected = {
            "./arbitrary filename.png": ImageConfig(
                main_prompts={ 'global prompt', 'local prompt' }, 
                tags={ Tag("global tag"), Tag("local tag") },
                flip_p=0.0
                ),
            "./sub/sub arbitrary filename.png": ImageConfig(
                main_prompts={ 'global prompt' }, 
                tags={ Tag("global tag") },
                flip_p=0.0
                ),
            "./sub/sidecar.png": ImageConfig(
                main_prompts={ 'global prompt', 'sidecar prompt' }, 
                tags={ Tag("global tag"), Tag("sidecar tag") },
                flip_p=0.0
                ),
            "./optfile/optfile.png": ImageConfig(
                main_prompts={ 'global prompt' }, 
                tags={ Tag("global tag") },
                flip_p=0.1234
                ),
            "./sub/sub2/sub3/xyz.png": ImageConfig(
                main_prompts={ 'global prompt' }, 
                tags={ Tag("global tag"), Tag("sub global tag") },
                flip_p=0.0
                )
        }
        self.assertEqual(expected, actual) 

    def test_can_load_dataset_from_json_manifest(self):
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
                    "main_prompt": [
                        "first caption",
                        "second caption"
                    ]
                }
            ]
            """))

        actual = Dataset.from_json("./manifest.json").image_configs
        expected = {
            "./stuff/image_1.jpg": ImageConfig( main_prompts={"default caption"} ),
            "./other/image_1.jpg": ImageConfig( main_prompts={"other caption"} ),
            "./other/image_2.jpg": ImageConfig(
                main_prompts={ "complex caption" },
                rating=0.1,
                max_caption_length=1000,
                tags={
                    Tag("including"),
                    Tag("weighted tag", 999.9)
                    }
                ),
            "./other/image_3.jpg": ImageConfig(
                main_prompts={ "first caption", "second caption" },
                multiply=2,
                flip_p=0.5,
                cond_dropout=0.01
                )
        }
        self.assertEqual(expected, actual)

    def test_dataset_can_produce_train_items(self):
        dataset = Dataset({
            "1.jpg": ImageConfig(
                multiply=2,
                flip_p=0.1,
                cond_dropout=0.01,
                main_prompts=["first caption","second caption"],
                rating = 1.1,
                max_caption_length=1024,
                tags=frozenset([
                    Tag("tag"),
                    Tag("tag_2", 2.0)
                ])),
            "2.jpg": ImageConfig( main_prompts="single caption")
        })

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