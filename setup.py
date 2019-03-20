#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='scaf',
    version='1.0.0',
    description='Semantic Change Analysis with Word Frequencies',
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords=['computational linguistics', 'semantic change analysis', 'change detection', 'word embeddings'],
    url='https://github.com/englhardt/scaf',
    license='MIT',
    author='Adrian Englhardt',
    author_email='adrian.englhardt@gmail.com',
    packages=find_packages(exclude='scaf.tests'),
    install_requires=[
        'numpy >= 1.13.1',
        'changepoint >= 0.1.1',
        'gensim == 2.3.0',
        'google-compute-engine >= 2.8.13',
        'scikit-learn >= 0.19.0',
        'future >= 0.16.0',
        'scipy >= 0.19.1',
        'pandas >= 0.20.3',
        'enum-compat >= 0.0.2'
    ],
    dependency_links=[
        "git+https://github.com/englhardt/changepoint.git#egg=changepoint-0.1.1"
    ],
    test_suite='nose.collector',
    tests_require=[
        'nose >= 1.3.7',
        'tox >= 2.7.0'
    ],
    entry_points={
        'console_scripts': ['scaf_emb_train=scaf.jobs.training:main',
                            'scaf_emb_eval=scaf.jobs.embedding_evaluation:main',
                            'scaf_build_ts=scaf.jobs.build_timeseries:main',
                            'scaf_prepare_store=scaf.jobs.prepare_store:main',
                            'scaf_change_detection=scaf.jobs.change_detection:main']
    },
    include_package_data=True,
    zip_safe=False
)
