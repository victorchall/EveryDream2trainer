python -m venv venv
call "venv\Scripts\activate.bat"
echo should be in venv here
cd .
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url "https://download.pytorch.org/whl/cu116"
pip install transformers==4.25.1
pip install diffusers[torch]==0.13.0
pip install pynvml==11.4.1
pip install bitsandbytes==0.35.0
git clone https://github.com/DeXtmL/bitsandbytes-win-prebuilt tmp/bnb_cache
pip install ftfy==6.1.1
pip install aiohttp==3.8.3
pip install tensorboard>=2.11.0
pip install protobuf==3.20.1
pip install wandb==0.13.6
pip install pyre-extensions==0.0.23
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
::pip install "xformers-0.0.15.dev0+affe4da.d20221212-cp38-cp38-win_amd64.whl" --force-reinstall
pip install pytorch-lightning==1.6.5
pip install OmegaConf==2.2.3
pip install numpy==1.23.5
pip install keyboard
pip install lion-pytorch
python utils/patch_bnb.py
python utils/get_yamls.py
GOTO :eof

:ERROR
echo Something blew up. Make sure Pyton 3.10.x is installed and in your PATH.

:eof
ECHO done
pause
