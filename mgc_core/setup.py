# Filename: setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

extra_compile_args = []
if sys.platform.startswith("win"):
    extra_compile_args = ["/O2", "/DNDEBUG"]
else:
    extra_compile_args = ["-O3", "-DNDEBUG", "-march=native"]

ext_modules = [
    Extension(
        "fastgeo",
        sources=["fastgeo_core.cpp"],
        include_dirs=[get_pybind_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

class BuildExt(build_ext):
    c_opts = {}
    def build_extensions(self):
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            if ct == "msvc":
                ext.extra_compile_args = ["/O2", "/DNDEBUG"]
            self.build_extension(ext)

setup(
    name="fastgeo",
    version="0.1.0",
    author="you",
    description="Fast geodesic distance using Djikstra's algorithm with edge costs",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
