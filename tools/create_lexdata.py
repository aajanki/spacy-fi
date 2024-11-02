"""Prepare the lexeme probability lookup table."""

import gzip
import json
import math
import re
import typer
from pathlib import Path
from itertools import islice

sensitive_email_pattern = re.compile(
    r'\.[A-Za-z0-9_%+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9-]|.@gmail\.[A-Za-z0-9_%+.-]+|.@yahoo\.[A-Za-z0-9_%+.-]+|.@hotmail\.[A-Za-z0-9_%+.-]+',
    re.IGNORECASE
)
etunimi_sukunimi_email_pattern = re.compile(
    r'^(?:etunimi\.sukunimi|etu\.sukunimi|sukunimi\.etunimi|firstname\.lastname)@',
    re.IGNORECASE
)


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


def read_freqs(full_loc, num_tokens):
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
        valid_tokens = (x for x in parsed if is_valid_token(x[1]))
        for freq, token in islice(valid_tokens, num_tokens):
            probs[token] = math.log(freq) - log_total
            remaining_freq -= freq

    # Our OOV estimate is the remaining probability mass distributed evenly on
    # the excluded word types.
    oov_prob = math.log(remaining_freq) - log_total - math.log(n - len(probs))

    return probs, oov_prob


def is_valid_token(token, max_token_length=50):
    # Skip too long tokens
    is_not_too_long = len(token) <= max_token_length

    # Redact email addresses that potentially contain person names
    is_sensitive_email = (
        sensitive_email_pattern.search(token) and
        not etunimi_sukunimi_email_pattern.search(token)
    )

    return is_not_too_long and not is_sensitive_email


def parse_token_line(line):
    freq, token = line.strip().split(' ', 1)
    return int(freq), token


if __name__ == '__main__':
    typer.run(main)
