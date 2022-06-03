import bz2
import sys
import typer
from pathlib import Path
from tqdm import tqdm
from fi import FinnishExtended


def main(input_file: Path, output_file: Path):
    tokenizer = create_tokenizer()

    with open_input(input_file) as inf, open_output(output_file) as outf:
        for i, line in enumerate(tqdm(inf)):
            line = line.rstrip('\n')
            tokens = (x.text for x in tokenizer(line))
            outf.write(' '.join(tokens))
            outf.write('\n')

            if i % 200000 == 0:
                # This will leak memory unless the tokenizer is re-created
                # periodically
                del tokenizer
                tokenizer = create_tokenizer()


def create_tokenizer():
    return FinnishExtended().tokenizer


def open_input(p: Path):
    if str(p) == '-':
        return sys.stdin
    elif p.suffix == '.bz2':
        return bz2.open(p, 'rt', encoding='utf-8')
    else:
        return open(p, 'r')


def open_output(p: Path):
    if str(p) == '-':
        return sys.stdout
    elif p.suffix == '.bz2':
        return bz2.open(p, 'wt', encoding='utf-8')
    else:
        return open(p, 'w')


if __name__ == '__main__':
    typer.run(main)
