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
    version='0.0.9',
    description='Python module to summarise a video into a collage.',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT License',
    url='https://github.com/luiscarlosgph/videosum',
    packages=[
        'videosum',
        'videosum._methods',
    ],
    package_dir={
        'videosum'          : 'src',
        'videosum._methods' : 'src/_methods',
    }, 
    install_requires = [
        'cython==0.29.37',
        'pandas==1.3.5',
        'numpy==1.25.2', 
        'opencv-python==4.7.0.72',
        'numba==0.58.1',
        #'faiss-gpu',
        'scikit-image==0.22.0',
        'scikit-learn==1.3.2',
        'scikit-learn-extra==0.3.0',
        'imageio_ffmpeg==0.4.9',
        'tqdm==4.66.1',
        'seaborn==0.13.0',
        'six==1.16.0',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    test_suite='test',
)
