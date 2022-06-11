import json
import typer
from itertools import islice
from pathlib import Path
from .io import open_input, open_output


def main(rawfile: Path, jsonlfile: Path, limit: int=100000):
    with open_input(rawfile) as inf, open_output(jsonlfile) as outf:
        for line in islice(inf, limit):
            line = line.strip()
            json.dump({'text': line}, outf, ensure_ascii=False)
            outf.write('\n')


if __name__ == '__main__':
    typer.run(main)
