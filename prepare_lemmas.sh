#!/bin/sh

set -eu

zcat data/frequencies/finnish_vocab.txt.gz \
    | python lemma/mine_lemma_rules.py 

zcat data/frequencies/finnish_vocab.txt.gz \
    | head -n 3000000 \
    | python lemma/create_lemma_lookup.py data/lemma/fi_lemma_rules.json \
    > data/lemma/fi_lemma_lookup.json
