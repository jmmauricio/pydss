#!/usr/bin/env python
# coding: utf-8


import os
import io
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test


# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("pydss", "__init__.py")) as fp:
    exec(fp.read(), version)


# https://docs.pytest.org/en/latest/goodpractices.html#manual-integration
class PyTest(test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest

        sys.exit(pytest.main(shlex.split(self.pytest_args)))

# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="pydss",
    version=version['__version__'],
    description="Python Distribution System Simulator",
    author="Juan Manuel Mauricio",
    author_email="jmmauricio@us.es",
    url="http://pydss.github.io/",
    download_url="https://github.com/jmmauricio/pydss",
    license="MIT",
    keywords=[
        "distribution system", "electric engineering"
    ],
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "matplotlib",
        "bokeh",
        "scipy",
        "pandas",
        "numba>=0.25",
    ],
    tests_require=[
        "coverage",
        "pytest-cov",
    ],
    extras_require={
        'dev': [
            "pep8",
            "mypy",
            "sphinx",
            "sphinx_rtd_theme",
            "nbsphinx",
            "ipython"
        ]
    },
    packages=find_packages('.'),
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'pydss = pydss.cli:main'
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    long_description=io.open('README.md', encoding='utf-8').read(),
    package_data={"poliastro": ['tests/*.py']},
    include_package_data=True,
    zip_safe=False,
    cmdclass={'test': PyTest},
)