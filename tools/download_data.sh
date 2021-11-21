#!/bin/sh

set -eu

mkdir -p data/raw

git submodule update --init

echo "Downloading frequency data"
mkdir -p data/raw/frequencies
wget --directory-prefix data/raw/frequencies http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz

echo "Downloading Wikipedia dump"
mkdir -p data/raw/wikipedia-fi-2017
wget --directory-prefix data/raw/wikipedia-fi-2017 https://korp.csc.fi/download/wikipedia-fi/wikipedia-fi-2017-src/wikipedia-fi-2017-src.zip

echo "Download pre-trained floret embeddings"
mkdir -p data/train/floret
wget --directory-prefix data/train/floret https://github.com/aajanki/floret-finnish/releases/download/release%2F1/fi-300-50k-minn3-maxn5-epoch5.floret.gz
