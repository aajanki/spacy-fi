import typer
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple
from fi import FinnishExtended
from .io import open_input, open_output


def main(input_dir: Path, output_dir: Path, threads: Optional[int] = 4):
    input_files = input_dir.iterdir()
    inputs_and_outputs = [(p, output_dir / uncompressed_file_name(p)) for p in input_files]

    with Pool(threads) as pool:
        for output_file in pool.imap_unordered(tokenize_file, inputs_and_outputs, chunksize=1):
            print(f'Wrote {output_file}')


def tokenize_file(paths: Tuple[Path, Path]):
    input_file, output_file = paths
    with open_input(input_file) as inf, open_output(output_file) as outf:
        tokenizer = create_tokenizer()

        for i, line in enumerate(inf):
            line = line.rstrip('\n')
            tokens = (x.text for x in tokenizer(line))
            outf.write(' '.join(tokens))
            outf.write('\n')

            if i % 200000 == 0:
                # This will leak memory unless the tokenizer is re-created
                # periodically
                del tokenizer
                tokenizer = create_tokenizer()

    return output_file


def create_tokenizer():
    return FinnishExtended().tokenizer


def uncompressed_file_name(input_file: Path):
    if input_file.name.endswith('.bz2'):
        return input_file.with_suffix('').name
    else:
        return input_file.name


if __name__ == '__main__':
    typer.run(main)
