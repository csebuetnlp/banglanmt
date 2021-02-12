#!/usr/bin/env python
from setuptools import setup

try:
    with open('README.md') as file:
        long_description = file.read()
except IOError:
    long_description = "missing"


setup(
    name='segmentation',
    data_files = [("", ["LICENSE.txt"])],
    packages = ['segmentation'],
    package_dir = {'segmentation':''},
    install_requires=['regex'],
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'segmenter = segmentation.segmenter:main'
        ]
    }
)
