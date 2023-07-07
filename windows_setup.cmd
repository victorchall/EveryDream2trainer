python -m venv venv
call "venv\Scripts\activate.bat"
echo should be in venv here
cd .
python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url "https://download.pytorch.org/whl/cu118"
pip install -U transformers==4.29.2
pip install -U diffusers[torch]==0.18.0
pip install pynvml==11.4.1
pip install -U https://github.com/victorchall/everydream-whls/raw/main/bitsandbytes-0.38.1-py2.py3-none-any.whl
pip install ftfy==6.1.1
pip install aiohttp==3.8.4
pip install tensorboard>=2.11.0
pip install protobuf==3.20.1
pip install wandb==0.15.3
pip install pyre-extensions==0.0.29
pip install -U xformers==0.0.20
pip install pytorch-lightning==1.6.5
pip install OmegaConf==2.2.3
pip install numpy==1.23.5
pip install lion-pytorch
pip install compel~=1.1.3
pip install dadaptation
pip install safetensors
pip install open-flamingo==2.0.0
python utils/get_yamls.py
GOTO :eof

:ERROR
echo Something blew up. Make sure Pyton 3.10.x is installed and in your PATH.

:eof
ECHO done
pause
