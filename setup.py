from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

# Define the C extension module
symnmf_module = Extension(
    'symnmf',
    sources=['symnmfmodule.c', 'symnmf.c'],
    include_dirs=[numpy.get_include()],

)

# Setup configuration
setup(
    name='symnmf',
    version='1.0',
    description='Symmetric Non-negative Matrix Factorization',
    ext_modules=[symnmf_module],
    cmdclass={'build_ext': build_ext},
)
