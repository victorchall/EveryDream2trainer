# EveryDream Trainer 2.0

Welcome to v2.0 of EveryDream trainer! Now with more Diffusers, faster, and even more features!

For the most up to date news and community discussions, please join us on Discord!

[![Discord!](.github/discord_sm.png)](https://discord.gg/AkUbqnDgYu)

If you find this tool useful, please consider subscribing to the project on Patreon or a one-time donation on Ko-fi.  Your donations keep this project alive as a free open source tool with ongoing enhancements.

[![Patreon](.github/patreon-medium-button.png)](https://www.patreon.com/everydream) or  [![Kofi](.github/kofibutton_sm.png)](https://ko-fi.com/everydream).

If you're coming from Dreambooth, please [read this](doc/NOTDREAMBOOTH.md) for an explanation of why EveryDream is not Dreambooth.

## Requirements

Windows 10/11, Linux (Ubuntu 20.04+ recommended), or use the linux Docker container

Python 3.10.x

Nvidia GPU with 11GB VRAM or more (note: 1080 Ti and 2080 Ti may require compiling xformers yourself)

16GB system RAM recommended minimum

Single GPU is currently supported

32GB of system RAM recommended for 50k+ training images, but may get away with sufficient swap file and 16GB

Ampere or newer 24GB+ (3090/A5000/4090, etc) recommended for 10k+ images

...Or use any computer with a web browser and run on Vast/Colab.  See [Cloud](#cloud) section below.

## Video tutorials

### [Basic setup and getting started](https://www.youtube.com/watch?v=OgpJK8SUW3c) 

Covers install, setup of base models, startning training, basic tweaking, and looking at your logs

### [Multiaspect and crop jitter explainer](https://www.youtube.com/watch?v=0xswM8QYFD0)


Behind the scenes look at how the trainer handles multiaspect and crop jitter

## Cloud/Docker

### [Free tier Google Colab notebook](https://colab.research.google.com/github/victorchall/EveryDream2trainer/blob/main/Train_Colab.ipynb)

### * [RunPod / Vast Instructions](/doc/CLOUD_SETUP.md)
#### * [Vast.ai Video Tutorial](https://www.youtube.com/watch?v=PKQesb4om9I)
#### [Runpod Video Tutorial](https://www.youtube.com/watch?v=XAULP-4hsnA)

### [Docker image link](https://github.com/victorchall/EveryDream2trainer/pkgs/container/everydream2trainer)

## Docs

[Setup and installation](doc/SETUP.md)

[Download and setup base models](doc/BASEMODELS.md) 

[Data Preparation](doc/DATA.md)

[Training](doc/TRAINING.md) - How to start training

[Troubleshooting](doc/TROUBLESHOOTING.md)

[Basic Tweaking](doc/TWEAKING.md) - Important args to understand to get started

[Advanced Tweaking](doc/ADVANCED_TWEAKING.md) and [Advanced Optimizer Tweaking](/doc/OPTIMIZER.md)

[Chaining training sessions](doc/CHAINING.md) - Modify training parameters by chaining training sessions together end to end

[Shuffling Tags](doc/SHUFFLING_TAGS.md)

[Data Balancing](doc/BALANCING.md) - Includes my small treatise on model "preservation" with additional ground truth data

[Logging](doc/LOGGING.md) 

[Validation](doc/VALIDATION.md) - Use a validation split on your data to see when you are overfitting and tune hyperparameters

[Captioning](doc/CAPTION_COG.md) - Llava, Cog, etc. to generate synthetic captioning (or [Old scripts for git/blip](doc/CAPTION.md))

[Plugins](doc/PLUGINS.md) - (beta) write your own plugins to execute arbitrary code during training

[Contributing](doc/CONTRIBUTING.md)

[Citations and references](doc/CITATIONS.md)

[Rectified Flow and Reflow](doc/ADVANCED_TWEAKING.md#rectified-flow-and-reflow) - Advanced training technique for potentially improved sample quality and faster inference.
