#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from fense import __version__

setup(
    name='fense',
    author="blmoistawinde",
    author_email="1840962220@qq.com",
    version=__version__,
    license='MIT',
    keywords='audio captioning, evaluation, transformers',
    url='https://github.com/blmoistawinde/fense',
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages = find_packages(),
    platforms=["all"],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
      ],
    install_requires=open("requirements.txt", encoding='utf-8').read().split('\n')
)
