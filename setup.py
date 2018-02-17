#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import tempfile
from Cython.Build import cythonize

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def has_library(compiler, libname):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True


class build_ext(_build_ext):

    def build_extension(self, ext):
        import numpy
        include_dirs = [numpy.get_include()]
        library_dirs = []
        libraries = []
        compile_flags = []
        link_flags = []

        if sys.platform == "darwin":
            compile_flags += ["-march=native", "-mmacosx-version-min=10.9"]
            link_flags += ["-march=native", "-mmacosx-version-min=10.9"]
            if has_flag(self.compiler, "-fopenmp"):
                # This is where Homebrew now installs OMP
                self.compiler.library_dirs += ["/usr/local/opt/llvm/lib"]
                library_dirs += ["/usr/local/opt/llvm/lib"]
                if has_library(self.compiler, "omp"):
                    libraries += ["omp"]
                compile_flags += ["-fopenmp"]
                link_flags += ["-fopenmp"]
        else:
            if has_library(self.compiler, "m"):
                libraries += ["m"]
            if has_flag(self.compiler, "-fopenmp"):
                if has_library(self.compiler, "omp"):
                    libraries += ["omp"]
                elif has_library(self.compiler, "gomp"):
                    libraries += ["gomp"]
                compile_flags += ["-fopenmp"]
                link_flags += ["-fopenmp"]

        # Update the extension
        ext.include_dirs += include_dirs
        ext.library_dirs += library_dirs
        ext.library_dirs += libraries
        ext.extra_compile_args += compile_flags
        ext.extra_link_args += link_flags

        _build_ext.build_extension(self, ext)


src = os.path.join("transit_periodogram", "transit_periodogram_impl.pyx")
ext = Extension("transit_periodogram.transit_periodogram_impl",
                sources=[src], )

setup(
    name="transit_periodogram",
    packages=["transit_periodogram"],
    ext_modules=[ext],
    setup_require=['pytest-runner'],
    test_require=['pytest', 'pytest-cov'],
    cmdclass=dict(build_ext=build_ext),
)
