"""Count token frequencies in a text file."""

import sys
import typer
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from .io import open_input


def main(input_path: Path):
    freqs = Counter()
    with open_input(input_path) as inf:
        for line in tqdm(inf, mininterval=0.5, smoothing=0.05):
            freqs.update(line.split())

    for token, freq in freqs.most_common():
        sys.stdout.write(f'{freq:9} {token}\n')


if __name__ == '__main__':
    typer.run(main)
