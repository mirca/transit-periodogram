#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension("transit_periodogram.transit_periodogram_impl",
                sources=["transit_periodogram/transit_periodogram_impl.pyx"],
                include_dirs=[numpy.get_include()])

setup(
    name="transit_periodogram",
    packages=["transit_periodogram"],
    ext_modules=cythonize([ext]),
)
