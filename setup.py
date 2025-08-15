from setuptools import find_packages
from distutils.core import setup

setup(
    name='cyan_rl_gym',
    version='1.0.0',
    author='Cyan',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='cyan@xxxx',
    description='Template RL environments for Cyan Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)
