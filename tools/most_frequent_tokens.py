import gzip
import typer
from pathlib import Path
from tqdm import tqdm
from fi import FinnishExtended


def main(freqs_loc: Path = typer.Argument(..., help='Path to the input word frequencies'),
         freqs_output_loc: Path = typer.Argument(..., help='Output path for the selected frequencies'),
         num_tokens: int = typer.Option(100000, help='Maximum number of vectors to include')):
    print(f'Selecting {num_tokens} most frequent tokens')
    selected_tokens = collect_frequencies(freqs_loc, num_tokens)
    save_frequencies(freqs_output_loc, selected_tokens)


def collect_frequencies(freqs_loc, num_tokens):
    tokenizer = FinnishExtended().tokenizer
    freqs = {}
    with gzip.open(freqs_loc, 'rt', encoding='utf-8') as infile:
        for line in tqdm(infile):
            freq_str, text = line.strip().split(' ', 1)
            f = int(freq_str)

            if f < 5:
                break

            doc = tokenizer(text)
            if len(text) == 1:
                tokens = [doc[0].text]
            else:
                tokens = [t.text for t in doc if len(t) > 1]

            for t in tokens:
                freqs[t] = freqs.get(t, 0) + f

    return [(x[1], x[0]) for x in sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:num_tokens]]


def save_frequencies(freqs_output_loc, freqs):
    with gzip.open(freqs_output_loc, 'wt', encoding='utf-8') as f:
        for freq, token in freqs:
            f.write(f'{freq:>8} {token}\n')


if __name__ == '__main__':
    typer.run(main)
