"""Prepare a lexical data file for spacy train."""

import gzip
import json
import math
import sys
import typer
from pathlib import Path


def main(
    full_vocabulary_path: Path = typer.Argument(..., help='Path to the full vocabulary'),
    input_vocabulary_path: Path = typer.Argument(..., help='Path to the input vocabulary')
):
    probs, oov_prob = read_freqs(full_vocabulary_path, input_vocabulary_path)
    out = sys.stdout

    header = {'lang': 'fi', 'settings': {'oov_prob': oov_prob}}
    write_json_line(header, out)
    for orth, p in probs.items():
        word_data = {'orth': orth, 'prob': p}
        write_json_line(word_data, out)


def read_freqs(full_loc, freq_loc):
    total = 0
    n = 0
    with gzip.open(full_loc, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            n = i + 1
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            total += freq
    log_total = math.log(total)

    probs = {}
    remaining_freq = total
    with gzip.open(freq_loc, 'rt', encoding='utf-8') as f:
        for line in f:
            freq, token = line.strip().split(' ', 1)
            freq = int(freq)
            probs[token] = math.log(freq) - log_total
            remaining_freq -= freq

    # Our OOV estimate is the remaining probability mass distributed evenly on
    # the excluded word types.
    oov_prob = math.log(remaining_freq) - log_total - math.log(n - len(probs))

    return probs, oov_prob


def write_json_line(obj, fp):
    json.dump(obj, fp=fp, ensure_ascii=False)
    fp.write('\n')


if __name__ == '__main__':
    typer.run(main)
