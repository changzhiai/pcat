# -*- coding: utf-8 -*-

from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    """Create and go to the directory"""
    prevdir = os.getcwd()
    try:
        os.makedirs(newdir)
    except OSError:
        pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

@contextmanager
def walk(newdir):
    """Only go to the directory"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)