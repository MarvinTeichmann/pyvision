Pyvision Tutorial
==================

This tutorial contains a minimalistic "Hello, World!" pyvision project. This setup can be used as template for your own projects.

Setup
------

Before using this tutorial, please install pyvision as discussed [here](../README.md), e.g. by running: `pip install git+https://github.com/MarvinTeichmann/pyvision.git`


Using pyvision
---------------

In order to use pyvision with the "Hello, World!" model run the following line, which will create a self-contained train_dir with a copy of all source files from which training can be started. 

`
python train.py configs/hello_world.json
`

Training from the created logdir can than be started using the command line interface: `pv2 train <logdir>`

In order to start training immediately try using:

```
python train.py configs/hello_world.json --train # or
python train.py configs/hello_world.json --debug # also sets the model in debug mode
```

Also try the flags `--bench` and `--name` to give your experiments a proper name. Now run

```
pv2 train --help # or
python train.py --help
```

In order to see a full list of tools available use: `pv2 --help`.

### Plotting

Running experiments is fun, but the real science starts with evaluation. So lets create some plots. First, lets run two experiements on our two GPUs:

```
python train.py configs/hello_world.json --train  --bench plotting --name run0 --gpus 0
python train.py configs/hello_world.json --train  --bench plotting --name run1 --gpus 1
```

This should create two logdirs under namely:

``
<LOGDIR0> := $PV_DIR_DATA/hello_world/plotting/run0_hello_world_<TIMESTAMP>
<LOGDIR1> := $PV_DIR_DATA/hello_world/plotting/run1_hello_world_<TIMESTAMP>
``

Now we can create a plot looking at the training performance of run0: `pv2 plot <LOGDIR0>`. In order to compare the performance of our two experiments run:

``
pv2 plot <LOGDIR0> <LOGDIR1>
``

Of course you need to replace `<LOGDIR0>` and `<LOGDIR1>` with our own logdirs.


Pyvision Interfaces
--------------------

Pyvision has three interfaces, which all can be used interchangeable to control your experiments:

1) The command-line interface (cli): `pv2`.
2) The self-contained logdir interface.
3) The python interface.

We have already seen the CLI in action. It is quite self-explanatory. Run `pv2 --help` and `pv2 train --help` to explore it further.   

For the second interface create an logdir using `python train.py configs/hello_world.json` and take a closer look at the created files. You will see, that the folder contains a copy of the entire model source code as well as additional files like `train.py`, `plot.py` and `model.py`. The file `train.py` is automatically generated and different from the `train.py` in the root folder of this project. Using this `python train.py` in the logdir can be used to start training directly from this folder. This is very useful, if the experiment should be copied to a different environoment like a cluster or a cloud computing resource. 

The third interface is implemented in `pyvision.tools` and is very useful to automatically generate a large amount of experiments, for example for hyperparameter tuning. A first example on howto use the python interface can be found in the [train.py](train.py) file of this tutorial project.


Create your own Project
------------------------

Training a "Hello, world!" model can become boring quite quickly. However this project can be a very good template and starting point for your own experiments. In order to do so, let's first take a closer look at the [config_file](configs/hello_world.json). This file is the key which controls the experiment. Hyperparameter for your model should be stored here. In addition to the hyperparameter every model should contain a `pyvision` section which tells `pyvision` where it finds the files for your experiment. Let's take a closer look:


```
    "pyvision": {
        "project_name": "hello_world",
        "entry_point": "../model/hello.py",
        "plotter": "../model/plot.py",
        "required_packages": [
            // source files which should be tracked
            "../model"
        ],
        "optional_packages": [
            // source files which may be tracked
            "../../pyvision",
            "../../mutils"
        ],
        "copy_required_packages": true,
        "copy_optional_packages": true
    },
```

The required attributes are:

1) `entry_point`: The file from where training is started. 
2) `plotter`: The file implementing plotting.
3) `project_name`: A string, will affect the logdir path
4) `required_packages`: A list of source folders closely tied to the project. Should contain the main model sourcedir.
5) `optional_packages`: Additional source code.
6) `copy_[req/opt]_packages`: Whether to copy the corresponding source to the logdir.


### Entry Point 

The parameter `entry_point` is a pointer to the code which is executed by pyvision. The corresponding python file needs to be structured in a certained way to take advantage of all pyvision functionalities. A minimalistic example is implemented in by [hello.py](model/hello.py). Pyvision expects to find a function with the following interface here:

```
def create_pyvision_model(conf, logdir, init_training=True, debug=False):   
    [...]
    return model
```

This function is given the `config` and `logdir` of the model and is expected to create a class object with the following attributes: 

1) `fit(self, max_epochs=None)`: When called train the model.   
2) `evaluate(self, epoch)`: When called evaluate the model.   
3) `load_from_logdir(self, logdir=None)`: When called load the model from logdir.

That is it! All you need to build a pyvision project. A minimalistic implementation of such a class can be found in [hello.py::PVControll](model/hello.py). One more advice: I highly recommend to implement training, evaluation and the model building in seperate modules and use high-level function calls from `PVControll` to imported python models. Otherwise the file can become very large quickly. Happy Coding!

### Logger und Plotting

TODO








