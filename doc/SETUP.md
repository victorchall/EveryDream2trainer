## For Runpod/Vast instructions see [Cloud Setup](/doc/CLOUD_SETUP.md)
## For Google Colab see [Train_Colab.ipynb](/Train_Colab.ipynb)

## Install Python

Install Python 3.10 from here if you do not already have Python 3.10.x installed.

https://www.python.org/downloads/release/python-3109/

https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe

Download and install Git from [git-scm.com](https://git-scm.com/).

or [Git for windows](https://gitforwindows.org/)

Make sure Python 3.10 shows on your command window:

    python --version

You should see ```Python 3.10.something```.  3.10.5, 3.10.9, etc.  It needs to be 3.10.x.

If you have Python 3.10.x installed but your command window shows another version (3.8.x, 3.9.x) ask for assistance in the discord.
...or you can try setting the path to the 310 binaries before running the windows_setup.cmd if you know what you're doing.

    SET PYTHON=C:\Python310\python.exe

*(you'll have to locate python 310 on your system on your own if you)*

## Clone this repo
Clone the repo from normal command line then change into the directory:

    git clone https://github.com/victorchall/EveryDream2trainer

Then change into the folder:

    cd EveryDream2trainer

## Windows

While still in the command window, run windows_setup.cmd to create your venv and install dependencies.

    windows_setup.cmd

Double check your python version again after setup by running these two commands:

    activate_venv.bat
    python --version

Again, this should show 3.10.x

## Local docker container

```sh
    docker compose up
```

And you can either get a shell via:
```sh
    docker exec -it everydream2trainer-docker-everydream2trainer-1 /bin/bash
```

Or go to your browser and hit `http://localhost:8888`. The web password is
`test1234` but you can change that in `docker-compose.yml`.

Your current source directory will be moutned to the Jupyter notebook.
