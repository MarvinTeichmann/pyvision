#!/usr/bin/env python

from setuptools import setup, find_packages

scripts = ['bin/pv2']


setup(name='pyvision',
      version='0.1',
      description='Several implementation of DenseCRF.',
      author='Marvin Teichmann',
      author_email='marvin.teichmann@googlemail.com',
      packages=find_packages(),
      scripts=scripts,
      package_data={'': ['*.lst']}
      )
