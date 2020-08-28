#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals

import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup

long_description = \
"""Finnish language model for SpaCy.

The model contains POS tagger, dependency parser, word vectors, noun
phrase extraction, token frequencies and a lemmatizer.
"""


def load_meta(fp):
    with io.open(fp, encoding='utf8') as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith('.'):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    return output


def data_files(data_dirs):
    output = []
    for d in data_dirs:
        output.extend(list_files(d))
    output.append('meta.json')
    return output


def list_requirements(meta):
    parent_package = meta.get('parent_package', 'spacy')
    requirements = [parent_package + meta['spacy_version']]
    if 'setup_requires' in meta:
        requirements += meta['setup_requires']
    if 'requirements' in meta:
        requirements += meta['requirements']
    return requirements


def setup_package():
    root = path.abspath(path.dirname(__file__))
    meta_path = path.join(root, 'meta.json')
    meta = load_meta(meta_path)
    model_name = 'spacy_{}_{}'.format(meta['lang'], meta['name'])
    model_dir = path.join(model_name, '{}_{}-{}'.format(meta['lang'], meta['name'], meta['version']))
    lookups_dir = path.join(model_name, 'lookups')

    copy(meta_path, model_name)
    copy(meta_path, model_dir)

    setup(
        name=model_name,
        description=meta['description'],
        long_description=long_description,
        author=meta['author'],
        author_email=meta['email'],
        url=meta['url'],
        version=meta['version'],
        license=meta['license'],
        packages=[model_name],
        package_data={model_name: data_files([model_dir, lookups_dir])},
        install_requires=list_requirements(meta),
        zip_safe=False,
        classifiers=[
            'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
            'Natural Language :: Finnish',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Text Processing',
        ],
    )


if __name__ == '__main__':
    setup_package()
