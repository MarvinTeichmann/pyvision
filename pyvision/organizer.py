"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import imp
import json
import logging

import shutil
from shutil import copyfile

from datetime import datetime
from functools import reduce

import numpy as np
import scipy as scp


from pyvision import utils as pvutils


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def change_value(config, key, new_value):

    key_list = key.split(".")
    keys, lastkey = key_list[:-1], key_list[-1]

    # Test whether value is present
    reduce(dict.__getitem__, key_list, config)

    reduce(dict.__getitem__, keys, config)[lastkey] = new_value


def get_logdir_name(project=None, bench=None,
                    cfg_file=None,
                    prefix=None, config=None,
                    timestamp=True):

    root_dir = os.path.join(os.environ['TV_DIR_RUNS'])

    if config is not None:
        project = config['pyvision']['project_name']

    if project is not None:
        root_dir = os.path.join(root_dir, project)

    if bench is not None:
        root_dir = os.path.join(root_dir, bench)

    if cfg_file is not None:
        json_name = cfg_file.split('/')[-1].replace('.json', '')
    else:
        json_name = 'unnamed'

    if prefix is not None:
        run_name = prefix + "_" + json_name
    else:
        run_name = json_name

    if timestamp:

        date = datetime.now().strftime('%Y_%m_%d_%H.%M')
        run_name = '%s_%s' % (run_name, date)

    logdir = os.path.join(root_dir, run_name)
    return logdir


def load_model_from_logdir(logdir, filewriter=False):
    config_file = os.path.join(logdir, 'config.json')
    main_script = os.path.join(logdir, 'model.py')
    source_dir = os.path.join(logdir, 'source')
    add_source = os.path.join(source_dir, 'additional_packages')

    logging.info("Loading Config file: {}".format(config_file))
    config = json.load(open(config_file))
    # TODO Make optional
    sys.path.insert(0, source_dir)
    sys.path.insert(1, add_source)

    # Create an output log file
    if filewriter:
        logfile = os.path.join(logdir, 'output.log')
        logging.info("All output will be written to: {}".format(logfile))
        pvutils.create_filewrite_handler(logfile, mode='a')

    m = imp.load_source('model', main_script)

    model = m.create_pyvision_model(conf=config, logdir=logdir)
    model.load_from_logdir()

    return model


def init_logdir(config, cfg_file, logdir):

    if config is None:
        config = json.load(open(cfg_file))

    logging.info("Initializing Logdir: {}".format(logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        logging.warning("Path exists: {}".format(logdir))
        logging.warning("Potentially overwriting existing model.")

    # Create an output log file
    logfile = os.path.join(logdir, 'output.log')
    # logging.info("All output will be written to: {}".format(logfile))
    pvutils.create_filewrite_handler(logfile)

    basedir = os.path.dirname(os.path.realpath(cfg_file))

    # Copy the main model file
    source_file = os.path.join(basedir, config['pyvision']['entry_point'])
    target_file = os.path.join(logdir, "model.py")
    copyfile(source_file, target_file)

    # Copy the plotting file
    source_file = os.path.join(basedir, config['pyvision']['plotter'])
    target_file = os.path.join(logdir, "plot.py")
    copyfile(source_file, target_file)

    # Save config file
    conffile = os.path.join(logdir, 'config.json')
    json.dump(config, open(conffile, 'w'), indent=4, sort_keys=True)

    package_dir = os.path.join(logdir, 'source')
    if config['pyvision']['copy_required_packages']:
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)
        for dir_name in config['pyvision']['required_packages']:
            src = os.path.join(basedir, dir_name)
            src = os.path.realpath(src)

            name = os.path.basename(dir_name)
            dst = os.path.join(package_dir, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    if config['pyvision']['copy_optional_packages']:
        opt_dir = os.path.join(package_dir, 'additional_packages')
        if not os.path.exists(opt_dir):
            os.mkdir(opt_dir)
        for dir_name in config['pyvision']['optional_packages']:
            src = os.path.join(basedir, dir_name)
            src = os.path.realpath(src)

            name = os.path.basename(dir_name)
            dst = os.path.join(opt_dir, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


if __name__ == '__main__':
    logging.info("Hello World.")
