"""Prepare a lexical data file for spacy train."""

import gzip
import math
import sys
import srsly
from preshed.counter import PreshCounter
from tqdm import tqdm


def main():
    infile, outfile = sys.argv[1:3]
    probs, oov_prob = read_freqs(infile)
    header = {'lang': 'fi', 'settings': {'oov_prob': oov_prob}}
    lex = [{'orth': orth, 'prob': p} for orth, p in probs.items()]
    data = [header] + lex
    srsly.write_jsonl(outfile, data)


def read_freqs(freq_loc, min_freq=100, max_length=100):
    counts = PreshCounter()
    total = 0
    n = 0
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            counts.inc(i + 1, freq)
            total += freq
            n += 1
    counts.smooth()
    log_total = math.log(total)

    probs = {}
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=n):
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            if freq >= min_freq and len(token) < max_length:
                smooth_count = counts.smoother(freq)
                probs[token] = math.log(smooth_count) - log_total

    oov_prob = math.log(counts.smoother(0)) - log_total
    return probs, oov_prob


if __name__ == '__main__':
    main()
