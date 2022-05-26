"""Count token frequencies in a text file."""

import gzip
import typer
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def main(tokenized_file: Path, output_file: Path):
    freqs = Counter()
    with tokenized_file.open() as inf:
        for line in tqdm(inf):
            tokens = line.strip('\n').split()
            freqs.update(tokens)

    with gzip.open(output_file, 'wt', encoding='utf-8') as outf:
        for token, freq in freqs.most_common():
            outf.write(f'{freq:9} {token}\n')


if __name__ == '__main__':
    typer.run(main)
