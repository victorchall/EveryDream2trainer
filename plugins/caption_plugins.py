from argparse import Namespace
from typing import List
import os
import re
import json
import logging
from colorama import Fore, Style
import importlib, pkgutil

class TestBase():
    def __init__(self):
        self.a = 1

    def __repr__(self) -> str:
        return f"TestBase: {self.a}"

class TestSub(TestBase):
    def __init__(self):
        super().__init__()
        self.b = 2

    def __repr__(self) -> str:
        return f"TestSub: {self.a}, {self.b}"

class PromptIdentityBase():
    """
    Base class for prompt alternation plugins, useful for captioning, etc.
    """
    def __init__(self, description: str="identity", key: str="indentity_plugin", fn: callable=None, args: Namespace=None):
        self.description = description
        #print(f"PromptIdentityPlugin: __init__ with fn: {fn}")
        if fn is None:
            fn = self._prompt_identity_from_args
            #print(f"{self.__class__}: fn is None, setting to self._prompt_identity_from_args")
        self.fn = fn
        self._key = key
        self.args = args
        #print(f"self._key: {self._key}")
    
    @property
    def key(self) -> str:
        return self._key

    def _prompt_identity_from_args(self, args: Namespace) -> str:
        #print("Wat")
        if "prompt" not in args:
            raise ValueError(f"prompt is required for prompt_identity_from_args")
        #print(f"prompt: {args.prompt}")
        #print(f"{type(args)}, type(prompt): {type(args.prompt)}")
        return args.prompt

    def __repr__(self) -> str:
        return f"Plugin Function: \"{self.key}\" - {self.description}"
    
    def __str__(self) -> str:
        return self.__repr__()

    def __call__(self, image_path, args: Namespace) -> str:
        #print(f"Calling {self.key} with image_path: {image_path}, args: {args}")
        args.image_path = image_path
        return self.fn(args)

    @staticmethod
    def _add_hint_to_prompt(hint: str, prompt: str) -> str:
        if "\{hint\}" in prompt:
            prompt = prompt.replace("\{hint\}", hint)
        else:
            prompt = f"Hint: {hint}\n{prompt}"
        return prompt

class HintFromFilename(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        super().__init__(key="hint_from_filename",
                         description="Add a hint to the prompt using the filename of the image (without extension)", 
                         fn=self._from_filename,
                         args=args)

    def _from_filename(self, args: Namespace) -> str:
        image_path = args.get("image_path", "")
        filename = os.path.splitext(image_path)[0]
        prompt = self._add_hint_to_prompt(filename, prompt)
        return prompt

class RemoveUsingCSV(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        super().__init__(key="remove_using_csv",
                         description="Removes whole word matches of the csv passed in from the prompt", 
                         fn=self._remove_using_csv,
                         args=args)

    def _filter_logic(self, prompt: str, filters: List[str]) -> str:
        # word boundary filter
        pattern = r'\b(?:' + '|'.join([re.escape(word) for word in filters]) + r')\b'

        result = re.sub(pattern, '', prompt)
        
        # fix up extra space and punctuation
        result = re.sub(r'\s{2,}', ' ', result)  # Remove extra spaces
        result = re.sub(r'\s([,.!?;])', r'\1', result)  # Fix punctuation and spaces
        
        return result.strip()

    def _remove_using_csv(self, args: Namespace) -> str:
        prompt = args.prompt
        csv = args.csv
        if len(csv) == 0:
            logging.error(f"** {Fore.RED}Error: csv is required for remove_using_csv{Style.RESET_ALL}")
        else:
            words = csv.split(",")
            for word in words:
                prompt = self._filter_logic(prompt, [word])
        return prompt

class HintFromLeafDirectory(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        super().__init__(key="from_leaf_directory",
                         description="Adds a hint to the prompt using the leaf directory name (last folder in path)", 
                         fn=self._from_leaf_directory,
                         args=args)

    def _from_leaf_directory(self, args:Namespace) -> str:
        image_path = args.image_path
        prompt = args.prompt
        leaf_folder_of_image = os.path.basename(os.path.dirname(image_path))
        return self._add_hint_to_prompt(leaf_folder_of_image, prompt)

class MetadataProvider():
    """ provides and caches metadata"""
    def __init__(self):
        self._datadict = {}

    def _from_metadata(self, args) -> dict:
        image_path = args.image_path
        prompt = args.get("prompt", "")
        metadata = self._get_metadata_dict(image_path)
        return f"metadata: {metadata}\n{prompt}"

    def _get_metadata_dict(self, metadata_path: str) -> dict:
        if not metadata_path in self._datadict:
            metadata_dirname = os.path.dirname(metadata_path)
            if not os.path.exists(metadata_path):
                logging.warning(f" metadata.json not found in {metadata_dirname}, ignoring{Style.RESET_ALL}")
                self._datadict[metadata_path] = {}
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self._datadict[metadata_path] = metadata

        return self._datadict[metadata_path]

class FromFolderMetadataJson(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        super().__init__(key="from_folder_metadata", 
                         description="Looks for metadata.json in the folder of the images and prefixes it to the prompt", 
                         fn=self._from_metadata_json,
                         args=args)
        self.metadata_provider = MetadataProvider()

    def _clean_metadata(self, metadata: dict, args) -> dict:
        if "remove_keys" in args:
            keys = args.remove_keys.split(",")
            logging.debug(f"Removing keys: {keys}")
            for key in keys:
                metadata.pop(key, None)
                logging.debug(f"Removed key: {key}")

    def _from_metadata_json(self, args:Namespace) -> dict:
        image_path = args.image_path
        image_dir = os.path.dirname(image_path)
        metadata_json_path = os.path.join(image_dir, "metadata.json")
        metadata = self.metadata_provider._get_metadata_dict(metadata_json_path)
        self._clean_metadata(metadata, args)
        metadata = json.dumps(metadata, indent=2)
        prompt = self._add_hint_to_prompt(f"metadata: {metadata}", args.prompt)
        return prompt

class TagsFromFolderMetadataJson(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        self.cache = {}
        super().__init__(key = "tags_from_metadata_json", 
                         description="Adds tags hint from metadata.json (in the samefolder as the image) to the prompt", 
                         fn=self._tags_from_metadata_json,
                         args=args)
        self.metadata_provider = MetadataProvider()

    def _tags_from_metadata_json(self, args:Namespace) -> str:
        image_path = args.image_path
        
        current_dir = os.path.dirname(image_path)
        metadata_json_path = os.path.join(current_dir, "metadata.json")
        self.metadata_provider._get_metadata_dict(metadata_json_path).get("tags", []) 

        prompt = args.prompt
        if len(tags) > 0:
            tags = ", ".join(tags)
            return self._add_hint_to_prompt(f"tags: {tags}", prompt)
        return prompt

class TitleAndTagsFromImageJson(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        super().__init__(key="title_and_tags_from_image_json",
                         description="Adds title and tags hint from metadata.json (in the samefolder as the image) to the prompt", 
                         fn=self._title_and_tags_from_metadata_json,
                         args=args)

    def _title_and_tags_from_metadata_json(self, args:Namespace) -> str:
        prompt = args.prompt
        logging.debug(f" {self.key}: prompt before: {prompt}")
        image_path = args.image_path
        current_dir = os.path.dirname(image_path)
        image_path_base = os.path.basename(image_path)
        image_path_without_extension = os.path.splitext(image_path_base)[0]
        candidate_json_path = os.path.join(current_dir, f"{image_path_without_extension}.json")

        if os.path.exists(candidate_json_path):
            with open(candidate_json_path, "r") as f:
                metadata = json.load(f)

        title = metadata.get("title", "").strip()
        hint = f"title: {title}" if len(title) > 0 else ""

        tags = metadata.get("tags", []) 
        tags = tags.split(",") if isinstance(tags, str) else tags # can be csv or list
        if len(tags) > 0:
            tags = ", ".join(tags)
            hint += f", tags: {tags}"

        prompt = self._add_hint_to_prompt(hint, prompt)
        logging.debug(f" {self.key}: prompt after: {prompt}")
        return prompt

class TitleAndTagsFromFolderMetadataJson(PromptIdentityBase):
    def __init__(self, args:Namespace=None):
        self.cache = {}
        super().__init__(key="title_and_tags_from_metadata_json",
                         description="Adds title and tags hint from metadata.json (in the samefolder as the image) to the prompt", 
                         fn=self._title_and_tags_from_metadata_json,
                         args=args)

    def _title_and_tags_from_metadata_json(self, args:Namespace) -> str:
        prompt = args.prompt
        logging.debug(f" {self.key}: prompt before: {prompt}")
        image_path = args.image_path
        current_dir = os.path.dirname(image_path)
        metadata_json_path = os.path.join(current_dir, "metadata.json")

        if metadata_json_path not in self.cache:
            if not os.path.exists(metadata_json_path):
                logging.error(f"** {Fore.RED}Error: metadata.json not found in {current_dir}, skippin prompt modification{Style.RESET_ALL}")
                return prompt
            with open(metadata_json_path, "r") as f:
                metadata = json.load(f)
            self.cache[metadata_json_path] = metadata

        title = self.cache[metadata_json_path].get("title", "").strip()
        hint = f"title: {title}" if len(title) > 0 else ""

        tags = self.cache[metadata_json_path].get("tags", []) 
        tags = tags.split(",") if isinstance(tags, str) else tags # can be csv or list
        if len(tags) > 0:
            tags = ", ".join(tags)
            hint += f", tags: {tags}"

        prompt = self._add_hint_to_prompt(hint, prompt)
        logging.debug(f" {self.key}: prompt after: {prompt}")
        return prompt

class TitleAndTagsFromGlobalMetadataJson(PromptIdentityBase):
    """
    Adds title and tags hint from global metadata json given by '--metadatafilename'
    Note: you could just put your metadata in the prompt instead of using this plugin, but perhaps useful?
    """
    def __init__(self, args:Namespace=None):
        self.cache = {}
        self.metadata_loaded = False
        super().__init__(key="title_and_tags_from_global_metadata_json",
                         description="Adds title and tags hint from global metadata json given by '--metadatafilename mydata/somefile.json'", 
                         fn=self._title_and_tags_from_global_metadata_json,
                         args=args)

    def _title_and_tags_from_global_metadata_json(self, image_path: str, **kwargs) -> str:
        prompt = kwargs.get("prompt", "")
        metadata_json_path = kwargs.get("metadata_json_path", "")

        if not self.metadata_loaded: # kinda sloppy but avoids me having to think about reworking init args
            if not os.path.exists(metadata_json_path):
                raise FileNotFoundError(f"metadata.json not found in {metadata_json_path}")
            with open(metadata_json_path, "r") as f:
                metadata = json.load(f)
            self.cache[metadata_json_path] = metadata
            self.metadata_loaded = True

        title = self.cache[metadata_json_path].get("title", "")
        hint = f"title: {title}"

        tags = self.cache[metadata_json_path].get("tags", []) 
        if len(tags) > 0:
            tags = ", ".join(tags)
            hint += f", tags: {tags}"

        return self._add_hint_to_prompt(hint, prompt)

def is_subclass_of_subclass(attribute, base_class, recursion_depth=5):
    if attribute.__module__ == base_class.__module__:
        if issubclass(attribute, base_class) and attribute is not base_class:
            return True

        if recursion_depth == 0:
            return False
        recursion_depth -= 1
        for base in attribute.__bases__:
            if is_subclass_of_subclass(base, base_class, recursion_depth):
                return True
    return False

def get_prompt_alteration_plugin_list() -> list:
    plugins = []   

    for finder, name, ispkg in pkgutil.iter_modules(["plugins"]):
        plugins_module_name = f"plugins.{name}"
        
        if plugins_module_name == "plugins.caption_plugins":
            module = importlib.import_module(plugins_module_name)

            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)

                if isinstance(attribute, type) \
                    and attribute.__module__ == module.__name__ \
                    and is_subclass_of_subclass(attribute, PromptIdentityBase, recursion_depth=5) \
                    and attribute is not PromptIdentityBase:
                    
                    plugins.append(attribute)
        #print(f"done checking plugins_module_name: {plugins_module_name}")
    return plugins

def load_prompt_alteration_plugin(plugin_key: str, args) -> callable:
    if plugin_key is not None:
        prompt_alteration_plugins = get_prompt_alteration_plugin_list()
        
        for prompt_plugin_cls in prompt_alteration_plugins:
            plugin_instance = prompt_plugin_cls(args)
            #print(f"prompt_plugin_cls: {prompt_plugin_cls}")
            #print(f"prompt_plugin_cls.key: {prompt_plugin_cls.key}")
            if plugin_key == plugin_instance.key:
                logging.info(f" **** Found plugin: {plugin_instance.key}")
                return plugin_instance
        raise ValueError(f"plugin_key: {plugin_key} not found in prompt_alteration_plugins")
    else:
        logging.info(f"No plugin specified")
        return PromptIdentityBase(args=args)
