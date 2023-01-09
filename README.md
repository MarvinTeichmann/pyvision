Pyvision
========

A collection of framework independend tools for computer vision in python. Check out the [Tutorial](tutorial) for an easy introduction.


Getting started
----------------

To install `pyvision` and `mutils` run:

```
pip install git+https://github.com/MarvinTeichmann/pyvision.git
```

Alternative, to install the package in development mode use:

```
git clone git@github.com:MarvinTeichmann/pyvision.git   
cd pyvision   
pip install -e .
```

Mutils: usefule functionalities
-------------------------------

Mutils contains utility functions I have been using over and over again.

#### mutils.json

The module `mutils.json` is a comment json implementation useful for reading and saving config files. It uses the same interface as numpy / torch for save and load:

```
from mutils import json

conf = json.load("example.conf") # Read a config file
json.save("save.conf", conf) # Entries are sorted and intended
```

#### mutils.image

The module `mutils.image` contains util function useful when working with images. Most useful are:

1) `image.show` : Plots one or multiple images
2) `image.normalize` : Normalizes an image into range [0, 1]

```
from mutils import image as mimage

mimage.show(img1, img2) # Plots to images

norm_img = mimage.normalize(img, whitening=False)
```

`mutils.image.normalize` is Idempotence, if `whitening = False`. That is `n(n(I)) = n(I)` if `n = mutils.image.normalize`. The `whitening` option is very useful for training, but make sure that you only use it once.


Pyvision: Organize your experiments
------------------------------------

Pyvision provides a framework and command line tools to organize your experiments. Please check out the Pyvision [Tutorial](tutorial). The main modules are:

1) `organizer`: To create and use self-contained logdirs
2) `logger`: Store important values like loss for each epoch of training
3) `plotter`: Plot and compare values.


Pyvision expects two environoment variabels to be set. To do so put the following lines into your .profile, .zsh_rc or bash_rc:

```
export PV_DIR_DATA="/scratch/ssdfs2/DATA/"
export PV_DIR_RUNS="/nfs/marvfs/RUNS"
```

In python you will be able to acess those variables using:

```
data_dir = os.environ["PV_DIR_DATA"]
run_dir = os.environ["PV_DIR_RUNS"]
```


Troubleshoot
--------------

On Mac OS install `hdf5` and other relevant libraries using `brew install hdf5 c-blosc lzo bzip2`. Find `hdf5` install dir using:

```
brew info -q hdf5 c-blosc lzo bzip2|grep '/opt/homebrew
```

and then run: (hdf5 version number may differ)

```

HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2_2/ pip install tables
```








