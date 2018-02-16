#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

src = os.path.join("transit_periodogram", "transit_periodogram_impl.pyx")
ext = Extension("transit_periodogram.transit_periodogram_impl",
                sources=[src],
                include_dirs=[numpy.get_include()])

setup(
    name="transit_periodogram",
    packages=["transit_periodogram"],
    ext_modules=cythonize([ext]),
    setup_require=['pytest-runner'],
    test_require=['pytest', 'pytest-cov']
)
