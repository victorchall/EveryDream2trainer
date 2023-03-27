Torch 2 support is extremely pre-alpha.  It is not recommended for use.

There is a bunch of triton error spam in logs, I've tried to suppress the errors but it's not working yet.

Clone a new copy, the VENV is not compatible and git will not manage the environment

`git clone https://github.com/victorchall/EveryDream2trainer ed2torch2`
`cd ed2torch2`
`git checkout torch2`

run normal install (`windows_setup.cmd` or build the docker container)

Try running training.

If your hyperparametr/grad scaler goes to a tiny or massive value its not working. I sugget using validation to make sure its actually working.

If you have problems, might try to install the latest xformers wheel from the github actions:

Download the Xformers 3.10 cu118 wheel for your system

https://github.com/facebookresearch/xformers/actions/runs/4501451442

linux: ubuntu-22.04-py3.10-torch2.0.0+cu118
https://github.com/facebookresearch/xformers/suites/11760994158/artifacts/613483176

win: windows-2019-py3.10-torch2.0.0+cu118
https://github.com/facebookresearch/xformers/suites/11760994158/artifacts/613483194

Save the .whl file to your everydream2trainer folder

activate your venv and pip install the wheel file

`pip install xformers-0.0.17.dev484-cp310-cp310-win_amd64.whl`