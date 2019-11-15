#!/bin/sh

set -eu

mkdir -p data/spacy
mkdir -p models
mkdir -p data/UD_Finnish-TDT-preprocessed
mkdir -p data/finer-data-preprocessed

# Convert data to the SpaCy format

for f in train dev test
do
    python tools/preprocess_UD-TDT.py \
	   < data/UD_Finnish-TDT/fi_tdt-ud-$f.conllu \
	   > data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-$f.conllu

    spacy convert --lang fi -n 6 data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-$f.conllu data/spacy
done

for f in digitoday.2014.train.csv digitoday.2014.dev.csv digitoday.2015.test.csv wikipedia.test.csv
do
    python tools/preprocess_finer.py < data/finer-data/data/$f > data/finer-data-preprocessed/$f

    spacy convert --lang fi -n 6 -c ner data/finer-data-preprocessed/$f data/spacy
done

# The training data must contain at least one of each NER tag,
# otherwise the NER model will not be initialized. Combine a small ner
# dataset to the TDT data.
cat nerdata/ner-sentences-small.conllu data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-train.conllu \
    > data/spacy/combined-train.conllu
spacy convert --lang fi -n 6 data/spacy/combined-train.conllu data/spacy

echo "Validating tagger and parser training and dev data"
python spacy_fi.py debug-data fi \
       data/spacy/combined-train.json \
       data/spacy/fi_tdt-ud-dev.json \
       --pipeline tagger,parser

echo "Validating NER training and dev data"
python spacy_fi.py debug-data fi \
       data/spacy/digitoday.2014.train.json \
       data/spacy/digitoday.2014.dev.json \
       --pipeline ner

# Training

rm -rf data/fi-experimental/*
spacy init-model fi data/fi-experimental \
      --model-name fi_experimental_web_md \
      --jsonl-loc data/lexdata.jsonl \
      --vectors-loc data/word2vec/finnish_parsebank_small.txt.gz \
      --prune-vectors 40000

rm -rf models1/*
python spacy_fi.py train fi models1 \
       data/spacy/combined-train.json \
       data/spacy/fi_tdt-ud-dev.json \
       --pipeline tagger,parser,ner \
       --vectors data/fi-experimental \
       --n-iter 50 \
       --n-early-stopping 4
rm -rf models1/model{?,??} models1/model-final

rm -rf models/*
python spacy_fi.py train fi models \
       data/spacy/digitoday.2014.train.json \
       data/spacy/digitoday.2014.dev.json \
       --pipeline ner \
       --vectors data/fi-experimental \
       --base-model models1/model-best \
       --n-iter 30

#python spacy_fi.py evaluate --displacy-path vis \
#       models/model-best data/spacy/digitoday.2015.test.json
