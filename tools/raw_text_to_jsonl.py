import json
import typer
from itertools import islice
from pathlib import Path

def main(rawfile: Path, jsonlfile: Path, limit: int=100000):
    with rawfile.open() as inf, jsonlfile.open('w') as outf:
        for line in islice(inf, limit):
            json.dump({'text': line}, outf, ensure_ascii=False)
            outf.write('\n')

if __name__ == '__main__':
    typer.run(main)
