# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:42:17 2021

@author: changai
"""

from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.15.0',
    'scipy>=1.1.0',
    'matplotlib>=2.2.0',
    'xlrd==1.2.0',
    'pandas>=1.2.4',
    'ase>=3.16.2',
]


setup(
    name="pcat",
    version='1.0',
    description="plot for catalysis",
    url="None",
    author="Changzhi Ai",
    author_email="changai@dtu.dk",
    install_required=install_requires,
    license="MIT",
    packages=find_packages()
)