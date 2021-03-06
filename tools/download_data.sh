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

echo "Downloading frequency data"
mkdir -p data/frequencies
wget --directory-prefix data/frequencies http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz
