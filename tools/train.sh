#!/bin/sh

set -euo pipefail

mkdir -p data/spacy
mkdir -p data/UD_Finnish-TDT-preprocessed
mkdir -p data/finer-data-preprocessed
mkdir -p models

## Preparing the training data ##

echo "Convert data to the SpaCy format"
rm -rf traindata/parser/*
for dataset in train dev test
do
    mkdir -p traindata/parser/$dataset
    python tools/preprocess_UD-TDT.py \
	   < data/UD_Finnish-TDT/fi_tdt-ud-$dataset.conllu \
	   > data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-$dataset.conllu

    spacy convert --lang fi -n 6 data/UD_Finnish-TDT-preprocessed/fi_tdt-ud-$dataset.conllu traindata/parser/$dataset
done

echo "Convert NER data"
for f in digitoday.2014.train.csv digitoday.2014.dev.csv digitoday.2015.test.csv wikipedia.test.csv
do
    python tools/preprocess_finer.py < data/finer-data/data/$f > data/finer-data-preprocessed/$f
done

rm -rf traindata/ner/*
mkdir -p traindata/ner/train
mkdir -p traindata/ner/dev
mkdir -p traindata/ner/test
spacy convert --lang fi -n 6 -c ner data/finer-data-preprocessed/digitoday.2014.train.csv traindata/ner/train
spacy convert --lang fi -n 6 -c ner data/finer-data-preprocessed/digitoday.2014.dev.csv traindata/ner/dev
spacy convert --lang fi -n 6 -c ner data/finer-data-preprocessed/digitoday.2015.test.csv traindata/ner/test
spacy convert --lang fi -n 6 -c ner data/finer-data-preprocessed/wikipedia.test.csv traindata/ner/test

echo "Validating tagger and parser training and dev data"
spacy debug config fi.cfg
spacy debug data fi.cfg

## Training ##

#spacy init labels fi.cfg labels

echo "Preparing vectors"
python tools/select_tokens.py --num-tokens 500000 \
       data/frequencies/finnish_vocab.txt.gz \
       data/word2vec/finnish_4B_parsebank_skgram.bin \
       data/frequencies/finnish_vocab_500k.txt.gz \
       data/word2vec/finnish_500k_parsebank.txt.gz

spacy init vectors fi \
      data/word2vec/finnish_500k_parsebank.txt.gz \
      data/vectors \
      --prune 20000 \
      --name fi_exp_web_md.vectors

echo "Preparing lexical data"
mkdir -p data/vocab
rm -rf data/vocab/*
python tools/create_lexdata.py \
       data/frequencies/finnish_vocab.txt.gz \
       data/frequencies/finnish_vocab_500k.txt.gz \
       > data/vocab/vocab-data.jsonl

echo "Training"
spacy train fi.cfg --output models/taggerparser


## Evaluate ##
echo "Evaluating"
spacy evaluate models/taggerparser/model-best \
      traindata/parser/test \
      --displacy-path evaluation_parses
