# Validation

*This documentation is incomplete.  Please feel free to contribute to it.*

Validation allows you to use a split of your data for evaluating your training progress.  

When training a specific class, setting aside a portion of the data for validation will allow you to see trend lines you cannot see when purely looking at loss of the training itself.

While loss on your training data should trend downward, if you set aside a validation set, you can see when your validation loss starts to trend upward.  This is a sign that you are overfitting.  You can then adjust your hyperparameters to reduce overfitting, such as reducing LR or reducing training epochs. 

## How to use validation

The `validation_config` option is a pointer to a JSON config file with settings for use in validation.  There is a default validation file `validation_default.json` in the repo root, but it is not used unless you specify it.  

CLI use:

    --validation_config validation_default.json

or in a config file:

    "validation_config": "validation_default.json"

## Logging and intepreting validation

Validation adds `loss/val` to your tensorboard logs.  This is the loss of the validation data.  Since this is separated from your training data, when it starts to trend upward you know you are overfitting.  

Additional notes are available here: https://github.com/victorchall/EveryDream2trainer/pull/36