from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Define the C extension module
symnmf_module = Extension(
    "symnmf_module",
    sources=["symnmfmodule.c", "symnmf.c"],
)

# Setup configuration
setup(
    name="symnmf_module",
    version="1.0",
    description="Symmetric Non-negative Matrix Factorization",
    ext_modules=[symnmf_module],
    cmdclass={"build_ext": build_ext},
)
