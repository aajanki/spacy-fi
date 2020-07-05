"""Prepare a lexical data file for spacy train."""

import gzip
import math
import plac
import sys
import srsly
from itertools import islice
from pathlib import Path
from preshed.counter import PreshCounter


@plac.annotations(
    full_vocabulary_path=('Path to the full vocabulary', 'positional', None, Path),
    input_vocabulary_path=('Path to the input vocabulary', 'positional', None, Path),
    settings_path=('Output path for the lexeme settings file', 'positional', None, Path),
    prob_path=('Output path to the prob data file', 'positional', None, Path),
)
def main(
    full_vocabulary_path,
    input_vocabulary_path,
    settings_path,
    prob_path
):
    probs, oov_prob = read_freqs(full_vocabulary_path, input_vocabulary_path)
    settings = {"oov_prob": oov_prob}
    srsly.write_json(settings_path, settings)
    srsly.write_json(prob_path, probs)


def read_freqs(full_loc, freq_loc):
    counts = PreshCounter()
    total = 0
    n = 0
    with gzip.open(full_loc, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            n = i + 1
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            counts.inc(n, freq)
            total += freq
    counts.smooth()
    log_total = math.log(total)

    probs = {}
    last_freq = 0
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        for line in f:
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            last_freq = freq
            smooth_count = counts.smoother(freq)
            probs[token] = math.log(smooth_count) - log_total

    oov_prob = math.log(counts.smoother(last_freq/2)) - log_total
    return probs, oov_prob


if __name__ == '__main__':
    plac.call(main)
