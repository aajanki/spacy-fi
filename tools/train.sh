#!/bin/bash

set -euo pipefail

if [ "${1:-}" == "--pretrain" ]; then
    DO_PRETRAIN=1
else
    DO_PRETRAIN=0
fi

## Preparing the training data ##

mkdir -p data/preprocessed/UD_Finnish-TDT
mkdir -p data/train
mkdir -p models

echo "Convert data to the SpaCy format"
rm -rf data/train/parser/*
for dataset in train dev test
do
    mkdir -p data/train/parser/$dataset
    python tools/preprocess_UD-TDT.py \
	   < submodules/UD_Finnish-TDT/fi_tdt-ud-$dataset.conllu \
	   > data/preprocessed/UD_Finnish-TDT/fi_tdt-ud-$dataset.conllu

    spacy convert --lang fi -n 6 data/preprocessed/UD_Finnish-TDT/fi_tdt-ud-$dataset.conllu data/train/parser/$dataset
done

echo "Preparing vectors"
mkdir -p data/train/frequencies
python -m tools.most_frequent_tokens --num-tokens 1000000 \
       data/raw/frequencies/finnish_vocab.txt.gz \
       data/train/frequencies/finnish_vocab_1M.txt.gz

spacy init vectors fi \
      data/train/floret/fi-300-50k-minn3-maxn5-epoch5.floret.gz \
      data/train/vectors \
      --mode floret \
      --name fi_exp_web_md.vectors

echo "Preparing lexical data"
mkdir -p data/train/vocab
rm -rf data/train/vocab/*
python tools/create_lexdata.py \
       data/raw/frequencies/finnish_vocab.txt.gz \
       data/train/frequencies/finnish_vocab_1M.txt.gz \
       > data/train/vocab/vocab-data.jsonl

echo "Validating tagger and parser training and dev data"
spacy debug config fi.cfg --code-path fi/fi.py
spacy debug data fi.cfg --code-path fi/fi.py


## Training ##

if [ $DO_PRETRAIN -ne 0 ]; then
    echo "Pretrain"
    rm -rf models/pretrain
    spacy pretrain fi.cfg models/pretrain --code fi/fi.py

    cp models/pretrain/models300.bin pretrain/weights.bin
    #cp models/pretrain/config.cfg pretrain
    #cp models/pretrain/log.jsonl pretrain
fi

echo "Training"
spacy train fi.cfg --output models/taggerparser --code fi/fi.py --paths.init_tok2vec pretrain/weights.bin


## Evaluate ##
echo "Evaluating on the dev set"
spacy evaluate models/taggerparser/model-best data/train/parser/dev --code fi/fi.py --output metrics/dev.json

echo "Evaluating on the test set"
spacy evaluate models/taggerparser/model-best data/train/parser/test --code fi/fi.py --output metrics/test.json
