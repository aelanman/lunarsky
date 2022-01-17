# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from setuptools import setup
import io


with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "lunarsky",
    "author": "Adam E. Lanman",
    "url": "https://github.com/aelanman/lunarsky",
    "download_url": "https://github.com/aelanman/lunarsky/archive/refs/tags/v0.1.1.tar.gz",
    "license": "BSD",
    "description": "Astropy support for selenocentric (Moon)"
    "reference frames and lunar surface observatories.",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"lunarsky": "lunarsky"},
    "packages": ["lunarsky", "lunarsky.tests"],
    "use_scm_version": {
        "root": ".",
        "relative_to": __file__,
        "version_scheme": "post-release",
        "local_scheme": "no-local-version",
        "write_to": "lunarsky/version.py",
    },
    "include_package_data": True,
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "setup_requires": ["pytest-runner", "setuptools_scm"],
    "install_requires": ["numpy>=1.15", "astropy>3.0", "spiceypy", "jplephem"],
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "astronomy moon spice",
}

if __name__ == "__main__":
    setup(**setup_args)
