#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()


setup(
    name="neuralpredictors",
    version="0.0.0rc3",
    description="Sinz Lab Neural System Identification Utilities",
    author="Sinz Lab",
    author_email="software@sinzlab.net",
    url="https://github.com/sinzlab/neuralpredictors",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
