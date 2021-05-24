# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()
    requirements = [i.strip() for i in requirements]

setup(
    name="rpunct",
    version="1.0.1",
    author="Daulet Nurmanbetov",
    author_email="daulet.nurmanbetov@gmail.com",
    description="An easy-to-use package to  restore punctuation of text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Felflare/rpunct",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)