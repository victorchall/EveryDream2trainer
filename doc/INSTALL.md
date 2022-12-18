# Installation

## Windows

* Open a normal windows command prompt and run `windows_setup.bat` from the command line.  
*Do **not** double click the file from Windows File Explorer*, you need the command window open.

* While that is running, download the official xformers windows wheel from this URL: 
https://github.com/facebookresearch/xformers/suites/9544395581/artifacts/454051141

* Unzip the xformers file to the EveryDream2 folder

* Check your command line window to make sure no errors occured.  If you have errors, please post them in the Discord and ask for assistance.

* Once the command line is done with no errors, paste this command into the command prompt: 

    `pip install xformers-0.0.15.dev0+303e613.d20221128-cp310-cp310-win_amd64.whl`

* When you want to train in the future after closing the command line, run `activate_venv.bat` from the command line to activate the virtual environment again. (hint: you can type `a` then press tab, then press enter)

## Next step

Read the documentation to setup your base models from which you will train.
[Base Model setup](doc/BASEMODELS.md)