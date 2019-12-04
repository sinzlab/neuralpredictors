#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()


setup(
    name="ml-utils",
    version="0.0.0",
    description="Neuroscience and Machine Learning at Sinz-Lab ",
    author="Fabian Sinz",
    author_email="fabian.sinz@uni-tuebingen.de",
    url="https://github.com/sinzlab/ml-utils",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
