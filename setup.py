from setuptools import setup, find_packages
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(THIS_DIR, 'README.rst')) as f:
    long_desc = f.read()

setup(
    name = 'idesolver',
    version = '0.1.0',
    author = 'Josh Karpel',
    author_email = 'josh.karpel@gmail.com',
    # license = 'Apache',
    # description = 'A Python library for running simulations and generating visualizations.',
    long_description = long_desc,
    url = 'https://github.com/JoshKarpel/idesolver',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
    ],
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        'numpy',
        'scipy',
    ],
)
