from setuptools import setup, find_packages
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(THIS_DIR, 'README.rst')) as f:
    long_desc = f.read()

setup(
    name = 'idesolver',
    version = '1.0.0',
    author = 'Josh Karpel',
    author_email = 'karpel@wisc.edu',
    maintainer = 'Josh Karpel',
    maintainer_email = 'karpel@wisc.edu',
    long_description = long_desc,
    url = 'https://github.com/JoshKarpel/idesolver',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
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
        'numpy',
        'scipy',
    ],
)
