"""Prepare the lexeme probability lookup table."""

import gzip
import json
import math
import typer
from pathlib import Path
from itertools import islice


def main(
    full_vocabulary_path: Path = typer.Argument(..., help='Path to the full vocabulary'),
    output_path: Path = typer.Argument(..., help='Name of the output directory'),
    num_tokens: int = typer.Argument(..., help='Number of tokens to include in the output file'),
):
    probs, oov_prob = read_freqs(full_vocabulary_path, num_tokens)

    prob_lookup = {}
    for orth, p in probs.items():
        prob_lookup[orth] = p

    settings_lookup = {'oov_prob': oov_prob}

    with open(output_path / 'lexeme_prob.json', 'w') as out:
        json.dump(prob_lookup, out, ensure_ascii=False)

    with open(output_path / 'lexeme_settings.json', 'w') as out:
        json.dump(settings_lookup, out, ensure_ascii=False)


def read_freqs(full_loc, num_tokens, max_token_length=50):
    total = 0
    n = 0
    with gzip.open(full_loc, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            n = i + 1
            freq, _ = parse_token_line(line)
            total += freq
    log_total = math.log(total)

    probs = {}
    remaining_freq = total
    with gzip.open(full_loc, 'rt', encoding='utf-8') as f:
        parsed = (parse_token_line(x) for x in f)
        valid_tokens = (x for x in parsed if len(x[1]) <= max_token_length)
        for freq, token in islice(valid_tokens, num_tokens):
            probs[token] = math.log(freq) - log_total
            remaining_freq -= freq

    # Our OOV estimate is the remaining probability mass distributed evenly on
    # the excluded word types.
    oov_prob = math.log(remaining_freq) - log_total - math.log(n - len(probs))

    return probs, oov_prob


def parse_token_line(x):
    freq, token = x.strip().split(' ', 1)
    freq = int(freq)
    return freq, token


if __name__ == '__main__':
    typer.run(main)
