#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import unittest

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='videosum',
    version='0.0.1',
    description='Python module to summarise a video into a collage.',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT License',
    url='https://github.com/luiscarlosgph/endoseg',
    packages=['videosum'],
    package_dir={'videosum' : 'src'}, 
    install_requires = ['numpy', 'opencv-python'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    #test_suite = 'tests',
)
