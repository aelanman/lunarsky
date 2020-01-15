# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from setuptools import setup
import glob
import os
import io
import json

#data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
#with open(os.path.join('pyuvsim', 'GIT_INFO'), 'w') as outfile:
#    json.dump(data, outfile)

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'lunarsky',
    'author': 'Adam E. Lanman',
    'url': 'https://github.com/aelanman/lunarsky',
    'license': 'BSD',
    'description': 'Astropy support for selenocentric (Moon) reference frames and lunar surface observatories.',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'lunarsky': 'lunarsky'},
    'packages': ['lunarsky', 'lunarsky.tests'],
    'version': '0.0.1',
    'include_package_data': True,
    'test_suite': 'pytest',
    'tests_require': ['pytest'],
    'setup_requires': ['pytest-runner'],
    'install_requires': ['numpy>=1.15', 'astropy>3.0', 'spiceypy', 'jplephem'],
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Astronomy'],
    'keywords': 'astronomy moon spice'
}

if __name__ == '__main__':
    setup(**setup_args)
