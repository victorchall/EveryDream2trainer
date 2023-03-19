# Contribution guide

Thank you for your interest in contributing to EveryDream! 

## Contributor License Agreement

Please review the [CLA](EveryDream_CLA.txt) before issuing a PR.  You will be asked on your first submission to post your agreement for any code changes. 

This is not required for simple documentation changes (i.e. spelling mistakes, etc.)

## Contributing code

EveryDream 2 trainer is a fairly complex piece of machinery with many options, and supports several runtime environments such as local Windows/Linux via venv, Docker/WSL, and Google Colab.  One of the primary challenges is to ensure nothing breaks across platforms.  EveryDream has users across all these platforms, and indeed, it is a bit of a chore keeping everything working, but one of the primary goals of the project to keep it as accessible as possible whether you have a home PC to use or are renting a cloud instance. 

The most important thing to do when contributing is to make sure to *run training* on your preferred platform with your changes.  A short training session and confirming that the first sample image or two after a couple hundred steps or so will ensure *most* functionality is working even if you can't test every possible runtime environment or combination of arguments.  While automated tests can help with identifying regression, there's no replacement for actually running the whole software package.  A quick 10-15 minute test on a small dataset is sufficient! 

**If you cannot test every possible platform, your contribution is still welcome.**  Please note waht you can and did test in your PR and we can work together to ensure it works across platforms, either by analysis for trivial changes, or help with testing.  Some changes are small isolated and may not require full testing across all platforms, but noting how you tested will help us ensure it works for everyone. 

**WIP** *link to a test data set will go here* **WIP**

## Code style and linting

**WIP** (there's a .pylint at least...)

## Documentation

Please update the appropriate document file in `/doc` for your changes.  If you are adding a new feature, please add a new section for users to read in order to understand how to use it, or if it is a significant feature, add a new document and link it from the main README.md or from the appropriate existing document.  

## A few questions to ask yourself before working on enhancements

There is no formation process for contributing to EveryDream, but please consider the following before submitting a PR:

* Consider if the change is general enough to be useful to others, or is more specific to your project.  Changes should provide value to a broad audience.  Sometimes specific project needs can be served by a script for your specific data instead of a change to the trainer behavior, for instance.

* Please be mindful of adding new primary CLI arguments.  New args should provide significant value to weigh the lengthening of the arg list.  The current list is already daunting, and the project is meant to remain at least *somewhat* accessible for a machine learning project. There may be ways to expose new functionality in other ways for advanced users without making primary CLI args more complex. 

* It's best to bring up any changes to default behavior in the Discord first. 

* If you add or update dependencies make sure to update the [Docker requirements](../docker/requirements.txt), [windows_setup.cmd](../windows_setup.cmd), and [Colab dependencies install cell](../Train_Colab.ipynb).  Please note that in your PR what platforms you were able to test or ask for help on Discord.

* Please consider checking in on the Discord #help channel after release to spot possible bugs encountered by users after your PR is merged.

### Running tests

**WIP** There is a small suite of automated unit tests. **WIP**