from __future__ import print_function

from setuptools import setup, find_packages
import os
import sys

if any((sys.version_info.major < 3, sys.version_info.major >= 3 and sys.version_info.minor < 6)):
    version_str = '{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    print('ERROR: IDESolver requires Python 3.6+, but you are trying to install it into Python {}'.format(version_str))
    sys.exit(1)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(THIS_DIR, 'README.rst')) as f:
    long_desc = f.read()

setup(
    name = 'idesolver',
    version = '1.0.2',
    author = 'Josh Karpel',
    author_email = 'karpel@wisc.edu',
    maintainer = 'Josh Karpel',
    maintainer_email = 'karpel@wisc.edu',
    long_description = long_desc,
    url = 'https://github.com/JoshKarpel/idesolver',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
    ],
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        'numpy>=1.13.0',
        'scipy>=1.0.0',
    ],
)
