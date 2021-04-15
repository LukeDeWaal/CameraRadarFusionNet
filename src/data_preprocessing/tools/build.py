from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
from shutil import copyfile
import os

ext_modules = [
    Extension(
        "radar2cam",
        ["radar2cam.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='compute_overlap',
    ext_modules=cythonize("compute_overlap.pyx"),
    compiler_directives={'language_level' : "3"},
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)

setup(
    name='radar2cam',
    ext_modules=cythonize(ext_modules),
    compiler_directives={'language_level' : "3"},
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)

direc = 'build/lib.linux-x86_64-3.7/src/data_preprocessing/tools'
for file in os.listdir(direc):
    copyfile(os.path.join(direc, file), os.path.join(os.path.dirname(__file__), file))