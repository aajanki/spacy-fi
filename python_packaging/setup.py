#!/usr/bin/env python
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
    model_name = f"spacy_{meta['lang']}_{meta['name']}"
    model_dir = path.join(model_name, f"{meta['lang']}_{meta['name']}-{meta['version']}")

    copy(meta_path, path.join(model_name))
    copy(meta_path, model_dir)

    setup(
        name=model_name,
        description=meta.get('description'),
        author=meta.get('author'),
        author_email=meta.get('email'),
        url=meta.get('url'),
        version=meta['version'],
        license=meta.get('license'),
        packages=[model_name],
        package_data={model_name: list_files(model_dir)},
        install_requires=list_requirements(meta),
        zip_safe=False,
        entry_points={'spacy_models': ['{m} = {m}'.format(m=model_name)]},
        long_description=long_description,
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Text Processing',
        ],
    )


if __name__ == '__main__':
    setup_package()
