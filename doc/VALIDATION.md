# Validation

Validation allows you to split data for evaluating your training progress.  

When training a specific class, setting aside a portion of the data for validation will allow you to see trend lines you cannot see when purely looking at loss of the training itself.

## How to use validation

The `validation_config` option in a pointer to a JSON config file with settings for use in validation.  There is a default validation file `validation_default.json` in the repo root.  

You can copy this file to a new file and edit it to your liking, then point to it with the `validation_config` option. 