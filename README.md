# Master Thesis Project

## The central idea

Learn Pathguiding using small diffusion networks

### Install

1. Clone Repository
2. Create the conda environment from the environment.yml
    2.1 No guarantee that it is complete, but should work, when in doubt use pip to install missing dependencies
3. update the submodules using `git submodule update --init --recursive` (unless already cloned with the `--recursive` flag)
4. go into each submodule and install it into conda env
    4.1 Follow the instructions in each, but for most running `python setup.py install` in a command line with the conda env activated should be enough


#### Scenes

Some test scenes are created through python dictionaries in code. 
The others are omitted from this repository due to size.
They can be downloaded from here https://mitsuba.readthedocs.io/en/stable/src/gallery.html.
Put them into the `assets/scenes/` directory