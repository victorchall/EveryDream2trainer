# RunPod

**[Runpod Video Tutorial](https://www.youtube.com/watch?v=XAULP-4hsnA)**

Click here -> [EveryDream2 template](https://runpod.io/console/deploy?template=cpl3xoknjz?ref=oko38cd0) to load a fully configured Docker image.  Both Tensorboard and Jupyter lab are automatically started for you and you can simply click the links to connect.

If you wish to sign up for Runpod, please consider using this [referral link](https://runpod.io?ref=oko38cd0) to help support the project.  2% of your spend is given back in the form of credits back to the project and costs you nothing.

You can also [enable full SSH support](https://www.runpod.io/blog/how-to-achieve-true-ssh-on-runpod) by setting the PUBLIC_KEY environment variable

# Vast.ai

**[Vast.ai Video Tutorial](https://www.youtube.com/watch?v=PKQesb4om9I)**

The EveryDream2 Docker image makes running [vast.ai](https://console.vast.ai/) fairly easy.

`ghcr.io/victorchall/everydream2trainer:main`

Watch the video for a full setup example.  Once the template is configured you can simply launch into it using any rented GPU instance by selecting the EveryDream2 docker template. 

Make sure to copy the IP:PORT to a new browser tab to connect to Tensorboard and Jupyter. You can see the ports by clicking the IP:PORT-RANGE on your instance once it is started.
![Config](vastai_ports.jpg)
The line with "6006/tcp" will be Tensorboard and the line with "8888/tcp" will be Jupyter. Click one to select, copy, then paste into a new browser tab.

## Password for Jupyter Lab (all platforms that use Docker)

The default password is `EveryDream`. This can be changed by editing environment variables or start parameters depending on what platform you use, or for local use, modify the docker run command.

# Instance concerns

## Bandwith

Make sure to select an instance with high bandwidth as you will need to wait to download your base model then later upload the finished checkpoint down to your own computer or up to Hugginface.  500mbps+ is good, closer to 1gbit is better.  If you are uploading a lot of training data or wish to download your finished checkpoints directly to your computer you may also want to make sure the instance is closer to your physical location for improved transfer speed.  You pay for rental while uploading and downloading, not just during training!

## GPU Selection

EveryDream2 requires a minimum 12GB Nvidia instance. 

Hosts such as Vast and Runpod offer 3090 and 4090 instances which are good choices.  The 3090 24 GB is a very good choice here, leaving plenty of room for running higher resolutions (768+) with a good batch size at a reasonable cost.  

As of writing, the 4090 is now going to run quite a bit faster (~60-80%) than the 3090 due to Torch2 and Cuda 11.8 support, but costs more than the 3090.  You will need to decide if it is cost effective when you go to rent something. 

Common major cloud providers like AWS and GCP offer the T4 16GB and A10G 24GB which are suitable.  A100 is generally overkill and not economical, and the 4090 may actually be faster unless you really need the 40/80 GB VRAM to run extremely high resolution training (1280+).  24GB cards can run 1024+ by enabling gradient checkpointing and using smaller batch sizes. 

If you plan on running a significant amount of training over the course of many months, purchasing your own 3090 may be another option instead of renting at anything, assuming your electricity prices are not a major concern.  However, renting may be a good entry point to see if the hobby interests you first.

I do not recommend V100 GPUs or any other older architectures (K80, Titan, etc).  Many of them will not support FP16 natively and are simply very slow.  Almost no consumer cards prior to 30xx series have enough VRAM. 

## Shutdown

Make sure to delete the instance when you are done.  Runpod and Vast use a trash icon for this.  Just stopping the instance isn't enough, and you pay pay for storage or rental until you completely delete it.

# Other Platforms

The Docker container should enable running on any host that supports using Docker containers, including GCP or AWS, and potentially with lifecycle management services such as GKE and ECS.  

Most people looking to use GCP or AWS will likely already understand how to manage instances, but as a warning, make sure you know how to manage the instance lifecycles so you don't end up with a surprise bill at the end of the month. Leaving an instance running all month can get expensive!
