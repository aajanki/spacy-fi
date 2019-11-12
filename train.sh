#!/bin/sh

set -eu

mkdir -p data/spacy
mkdir -p models
mkdir -p data/UD_Finnish-TDT-preprocessed

# Preparing the data

for f in train dev test
do
    python tools/preprocess_UD-TDT.py \
	   < data/UD_Finnish-TDT/fi_tdt-ud-$f.conllu \
	   > data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-$f.conllu
done

spacy convert --lang fi -n 6 data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-train.conllu data/spacy
spacy convert --lang fi -n 6 data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-dev.conllu data/spacy

python spacy_fi.py debug-data fi \
       data/spacy/fi_tdt-ud-train.json \
       data/spacy/fi_tdt-ud-dev.json \
       --pipeline tagger,parser

# Training

rm -rf data/fi-experimental/*
spacy init-model fi data/fi-experimental \
      --model-name fi_experimental_web_md \
      --jsonl-loc data/lexdata.jsonl \
      --vectors-loc data/word2vec/finnish_parsebank_small.txt.gz \
      --prune-vectors 40000

rm -rf models/*
python spacy_fi.py train fi models \
       data/spacy/fi_tdt-ud-train.json \
       data/spacy/fi_tdt-ud-dev.json \
       --pipeline tagger,parser \
       --vectors data/fi-experimental
