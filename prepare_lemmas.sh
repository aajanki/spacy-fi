#!/bin/sh

set -eu

zcat data/frequencies/finnish_vocab.txt.gz \
    | head -n 4000000 \
    | python lemma/create_lemma_lookup.py \
    > data/lemma/fi_lemma_lookup4.json
