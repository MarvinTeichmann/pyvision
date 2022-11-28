Pyvision Tutorial
==================

This tutorial contains a minimalistic `hello world` pyvision project. This setup can be used as template for your own projects.

Setup
------

Before using this Tutorial, please install as discussed [here](../README.md), e.g. by running: 

`
pip install git+https://github.com/MarvinTeichmann/pyvision.git
`


Using pyvision
---------------

In order to use pyvision with the hello world model run: `python train.py configs/hello_world.json`. 

This will create a self-contained train_dir with a copy of all source files from which training can be started. 


In order to start training immediately also try running:

```
python train.py configs/hello_world.json --train
python train.py configs/hello_world.json --debug
```

Also try the flags `--bench` and `--name` to give your experiments a proper name. Now run

```
pv2
python train.py -h
```

