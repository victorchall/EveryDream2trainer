# EveryDream Trainer 2.0

Welcome to v2.0 of EveryDream trainer! Now with more diffusers and even more features!

Please join us on Discord! https://discord.gg/uheqxU6sXN

If you find this tool useful, please consider subscribing to the project on [Patreon](https://www.patreon.com/everydream) or a one-time donation at [Ko-fi](https://ko-fi.com/everydream).

## Video tutorials

### [Basic setup and getting started](https://www.youtube.com/watch?v=OgpJK8SUW3c) 

Covers install, setup of base models, startning training, basic tweaking, and looking at your logs
### [Multiaspect and crop jitter](https://www.youtube.com/watch?v=0xswM8QYFD0)

Behind the scenes look at how the trainer handles multiaspect and crop jitter

### Companion tools repo

Make sure to check out the [tools repo](https://github.com/victorchall/EveryDream), it has a grab bag of scripts to help with your data curation prior to training.  It has automatic bulk BLIP captioning for BLIP, script to web scrape based on Laion data files, script to rename generic pronouns to proper names or append artist tags to your captions, etc. 

## Docs

[Setup and installation](doc/SETUP.md)

[Download and setup base models](doc/BASEMODELS.md) 

[Data Preparation](doc/DATA.md)

[Training](doc/TRAINING.md) - How to start training

[Basic Tweaking](doc/TWEAKING.md) - Important args to understand to get started

[Logging](doc/LOGGING.md) 

[Advanced Tweaking](doc/ATWEAKING.md) - More stuff to tweak once you are comfortable

[Chaining training sessions](doc/CHAINING.md) - Modify training parameters by chaining training sessions together end to end

[Shuffling Tags](doc/SHUFFLING_TAGS.md)

[Data Balancing](doc/BALANCING.md) - Includes my small treatise on model preservation with ground truth data

[Validation](doc/VALIDATION.md) - Use a validation split on your data to see when you are overfitting and tune hyperparameters

## Cloud

[Free tier Google Colab notebook](https://colab.research.google.com/github/victorchall/EveryDream2trainer/blob/main/Train_Colab.ipynb)

