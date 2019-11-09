#!/bin/sh

set -eu

mkdir -p data/spacy
mkdir -p models

spacy convert --lang fi -n 2 data/UD_Finnish-TDT/fi_tdt-ud-train.conllu data/spacy
spacy convert --lang fi -n 2 data/UD_Finnish-TDT/fi_tdt-ud-dev.conllu data/spacy

#spacy init-model fi data/fi-experimental

python spacy_fi.py debug-data fi data/spacy/fi_tdt-ud-train.json data/spacy/fi_tdt-ud-dev.json --pipeline tagger,parser

rm -rf models/*
python spacy_fi.py train fi models data/spacy/fi_tdt-ud-train.json data/spacy/fi_tdt-ud-dev.json --pipeline tagger,parser
