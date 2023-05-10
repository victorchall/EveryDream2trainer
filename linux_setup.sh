python3 -m venv venv
source venv/bin/activate
echo should be in venv here
cd .
python3 -m pip install --upgrade pip
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
pip3 install transformers==4.27.1
pip3 install diffusers[torch]==0.13.0
pip3 install pynvml==11.4.1
pip3 install bitsandbytes==0.35.0
git clone https://github.com/DeXtmL/bitsandbytes-win-prebuilt tmp/bnb_cache
pip3 install ftfy==6.1.1
pip3 install aiohttp==3.8.3
pip3 install tensorboard>=2.11.0
pip3 install protobuf==3.20.1
pip3 install wandb==0.14.0
pip3 install pyre-extensions==0.0.23
pip3 install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl
#pip3 install "xformers-0.0.15.dev0+affe4da.d20221212-cp38-cp38-win_amd64.whl" --force-reinstall
pip3 install pytorch-lightning==1.6.5
pip3 install OmegaConf==2.2.3
pip3 install numpy==1.23.5
pip3 install keyboard
pip3 install lion-pytorch
pip3 install compel~=1.1.3
python3 utils/patch_bnb.py
python3 utils/get_yamls.py
