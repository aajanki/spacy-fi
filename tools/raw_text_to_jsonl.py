import bz2
import json
import sys
import typer
from itertools import islice
from pathlib import Path

def main(rawfile: Path, jsonlfile: Path, limit: int=100000):
    with open_input(rawfile) as inf, jsonlfile.open('w') as outf:
        for line in islice(inf, limit):
            line = line.strip()
            json.dump({'text': line}, outf, ensure_ascii=False)
            outf.write('\n')


def open_input(p: Path):
    if str(p) == '-':
        return sys.stdin
    elif p.suffix == '.bz2':
        return bz2.open(p, 'rt', encoding='utf-8')
    else:
        return open(p, 'r')


if __name__ == '__main__':
    typer.run(main)
