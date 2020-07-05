#!/bin/sh

set -eu

mkdir -p data/spacy
mkdir -p data/UD_Finnish-TDT-preprocessed
mkdir -p data/finer-data-preprocessed
mkdir -p models

echo "Convert data to the SpaCy format"
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

echo "Validating tagger and parser training and dev data"
python spacy_fi.py debug-data fi \
    data/spacy/fi_tdt-ud-train.json \
    data/spacy/fi_tdt-ud-dev.json \
    --pipeline tagger,parser \
    --tag-map-path fi/tag_map.json

echo "Validating NER training and dev data"
python spacy_fi.py debug-data fi \
    data/spacy/digitoday.2014.train.json \
    data/spacy/digitoday.2014.dev.json \
    --pipeline ner \
    --tag-map-path fi/tag_map.json

## Training ##

echo "Preparing lexical data"
python tools/select_tokens.py -n 500000 \
       data/frequencies/finnish_vocab.txt.gz \
       data/word2vec/finnish_4B_parsebank_skgram.bin \
       data/frequencies/finnish_vocab_500k.txt.gz \
       data/word2vec/finnish_500k_parsebank.txt.gz

mkdir -p data/lexeme
rm -rf data/lexeme/*
python tools/create_lexdata.py \
       data/frequencies/finnish_vocab.txt.gz \
       data/frequencies/finnish_vocab_500k.txt.gz \
       data/lexeme/fi_lexeme_settings.json \
       data/lexeme/fi_lexeme_prob.json

echo "Training"
rm -rf data/fi-experimental/*
python spacy_fi.py init-model fi data/fi-experimental \
    --model-name experimental_web_md \
    --vectors-loc data/word2vec/finnish_500k_parsebank.txt.gz \
    --prune-vectors 20000

# tagger and parser model
rm -rf models/taggerparser/*
python -W ignore:"[W033]" spacy_fi.py train fi \
    models/taggerparser \
    data/spacy/fi_tdt-ud-train.json \
    data/spacy/fi_tdt-ud-dev.json \
    --tag-map-path fi/tag_map.json \
    --pipeline tagger,parser \
    --vectors data/fi-experimental \
    --n-iter 60
rm -rf models/taggerparser/model{?,??} models/taggerparser/model-final

# NER model
rm -rf models/ner/*
python -W ignore:"[W033]" spacy_fi.py train fi \
    models/ner \
    data/spacy/digitoday.2014.train.json \
    data/spacy/digitoday.2014.dev.json \
    --tag-map-path fi/tag_map.json \
    --pipeline ner \
    --vectors data/fi-experimental \
    --n-iter 30
rm -rf models/ner/model{?,??} models/ner/model-final

# Merge POS tagger and NER models
python tools/mergemodels.py models/taggerparser/model-best models/ner/model-best models/merged

## Evaluate ##

python spacy_fi.py evaluate models/merged data/spacy/fi_tdt-ud-dev.json
python spacy_fi.py evaluate models/merged data/spacy/digitoday.2014.dev.json
