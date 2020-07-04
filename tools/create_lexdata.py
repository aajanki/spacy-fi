"""Prepare a lexical data file for spacy train."""

import gzip
import math
import plac
import sys
import srsly
from itertools import islice
from pathlib import Path
from preshed.counter import PreshCounter
from tqdm import tqdm


@plac.annotations(
    input_path=('Path to the input vocabulary file', 'positional', None, Path),
    settings_path=('Output path for the lexeme settings file', 'positional', None, Path),
    prob_path=('Output path to the prob data file', 'positional', None, Path),
    token_limit=('Maximum number of tokens to include in the output', 'option', 'n', int)
)
def main(
    input_path,
    settings_path,
    prob_path,
    token_limit=1000000
):
    probs, oov_prob = read_freqs(input_path, token_limit)
    settings = {"oov_prob": oov_prob}
    srsly.write_json(settings_path, settings)
    srsly.write_json(prob_path, probs)


def read_freqs(freq_loc, token_limit, max_length=100):
    counts = PreshCounter()
    total = 0
    n = 0
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            n = i + 1
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            counts.inc(n, freq)
            total += freq
    counts.smooth()
    log_total = math.log(total)

    probs = {}
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        num_lines = min(n, token_limit)
        for line in tqdm(islice(f, num_lines), total=num_lines):
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            if len(token) < max_length:
                smooth_count = counts.smoother(freq)
                probs[token] = math.log(smooth_count) - log_total

    oov_prob = math.log(counts.smoother(0)) - log_total
    return probs, oov_prob


if __name__ == '__main__':
    plac.call(main)
