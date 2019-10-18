from __future__ import print_function

import os
import sys
import re
from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

if any(
    (
        sys.version_info.major < 3,
        sys.version_info.major >= 3 and sys.version_info.minor < 6,
    )
):
    version_str = "{}.{}.{}".format(
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro
    )
    print(
        "ERROR: IDESolver requires Python 3.6+, but you are trying to install it into Python {}".format(
            version_str
        )
    )
    sys.exit(1)

with open(os.path.join(THIS_DIR, "README.rst")) as f:
    long_desc = f.read()


def find_version():
    """Grab the version out of idesolver/version.py without importing it."""
    version_file_text = (THIS_DIR / 'idesolver' / 'version.py').read_text()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file_text,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name = "idesolver",
    version = find_version(),
    author = "Josh Karpel",
    author_email = "karpel@wisc.edu",
    maintainer = "Josh Karpel",
    maintainer_email = "karpel@wisc.edu",
    short_description = "A general purpose iterative numeric integro-differential equation (IDE) solver",
    long_description = long_desc,
    url = "https://github.com/JoshKarpel/idesolver",
    license = "GPL v3.0",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    packages = find_packages(),
    install_requires = Path('requirements.txt').read_text().splitlines(),
)
