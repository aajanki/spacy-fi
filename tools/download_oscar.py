import re
import typer
from itertools import islice
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def main(
    subset: str,
    max_texts: int,
    output_file: Path
):
    dataset = load_dataset("oscar", subset, split="train", streaming=True)
    with open(output_file, 'w') as outf:
        for i, text in enumerate(tqdm(islice(lines(dataset), max_texts), total=max_texts)):
            text = re.sub('\s+', ' ', text)
            outf.write(text)
            outf.write('\n')


def lines(dataset):
    for x in iter(dataset):
        text = x['text'].strip()
        if text:
            yield text


if __name__ == '__main__':
    typer.run(main)
