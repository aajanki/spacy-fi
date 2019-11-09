#!/bin/sh

set -eu

mkdir -p data

git clone --branch r2.4 --single-branch --depth 1 https://github.com/UniversalDependencies/UD_Finnish-TDT data/UD_Finnish-TDT

mkdir -p data/word2vec
wget --directory-prefix data/word2vec http://dl.turkunlp.org/finnish-embeddings/finnish_4B_parsebank_skgram.bin
python tools/word2vec_bin2txt.py data/word2vec/finnish_4B_parsebank_skgram.bin \
       data/word2vec/finnish_4B_parsebank_skgram.txt.gz

mkdir -p data/frequencies
wget --directory-prefix data/frequencies http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz
python tools/create_lexdata.py \
       data/frequencies/finnish_vocab.txt.gz \
       data/lexdata.jsonl
