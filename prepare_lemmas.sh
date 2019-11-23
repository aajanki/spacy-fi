#!/bin/sh

set -eu

mkdir -p data/lemma

zcat data/frequencies/finnish_vocab.txt.gz \
    | head -n 4000000 \
    | python lemma/create_lemma_lookup.py \
    > data/lemma/fi_lemma_lookup.json

zcat data/frequencies/finnish_vocab.txt.gz \
    | head -n 4000000 \
    | python lemma/mine_lemma_rules.py
