#!/bin/sh

set -eu

mkdir -p data

echo "Downloading UD_Finnish-TDT"
git clone --branch r2.4 --single-branch --depth 1 https://github.com/UniversalDependencies/UD_Finnish-TDT data/UD_Finnish-TDT

echo "Downloading finer-data"
git clone https://github.com/mpsilfve/finer-data.git data/finer-data

echo "Downloading word vectors"
mkdir -p data/word2vec
wget --directory-prefix data/word2vec http://dl.turkunlp.org/finnish-embeddings/finnish_4B_parsebank_skgram.bin
python tools/word2vec_bin2txt.py data/word2vec/finnish_4B_parsebank_skgram.bin \
       data/word2vec/finnish_4B_parsebank_skgram.txt.gz
zcat data/word2vec/finnish_4B_parsebank_skgram.txt.gz | head -n 400001 | gzip \
        > data/word2vec/finnish_parsebank_small.txt.gz

echo "Downloading frequency data"
mkdir -p data/frequencies
wget --directory-prefix data/frequencies http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz

echo "Preparing lexical data"
python tools/create_lexdata.py \
       data/frequencies/finnish_vocab.txt.gz \
       data/lexdata.jsonl

echo "Preparing lemma lookups"
python tools/create_lemma_lookup.py \
       data/frequencies/finnish_vocab.txt.gz \
       data/fi_lemma_lookup.json
