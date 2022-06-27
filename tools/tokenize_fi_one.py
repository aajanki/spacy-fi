import typer
from pathlib import Path
from .tokenize_fi import tokenize_file


def main(input_file: Path, output_file: Path):
    tokenize_file((input_file, output_file))


if __name__ == '__main__':
    typer.run(main)
