import gzip
import re
import typer
from itertools import islice
from pathlib import Path
from spacy.lang.fi import Finnish
from gensim.models import KeyedVectors
from tqdm import tqdm


def main(freqs_loc: Path = typer.Argument(..., help='Path to the input word frequencies'),
         vectors_loc: Path = typer.Argument(..., help='Path to the input word2vec binary file'),
         freqs_output_loc: Path = typer.Argument(..., help='Output path for the selected frequencies'),
         vectors_output_loc: Path = typer.Argument(..., help='Output path for the word2vec text file'),
         num_tokens: int = typer.Option(100000, help='Maximum number of vectors to include')):
    print(f'Selecting {num_tokens} most frequent tokens')
    selected_tokens = select_tokens(freqs_loc, num_tokens)
    save_frequencies(freqs_output_loc, selected_tokens)

    print('Collecting vectors')
    # Common tokens not included in finnish_vocab.txt.gz
    extra_tokens = ['−', '—', '…', '<', '>']
    tokens = [t[1] for t in selected_tokens]
    tokens = tokens + [t for t in extra_tokens if t not in tokens]
    output_wv = select_vectors(vectors_loc, tokens)
    output_wv.save_word2vec_format(vectors_output_loc, binary=False)


def select_tokens(freqs_loc, num_tokens):
    tokenizer = Finnish().tokenizer
    freqs = (x.strip().split(' ', 1) for x in gzip.open(freqs_loc, 'rt', encoding='utf-8').readlines())
    freqs2 = (x for x in freqs if is_valid_token(tokenizer, x[1]))
    return list(tqdm(islice(freqs2, num_tokens), total=num_tokens))


def save_frequencies(freqs_output_loc, freqs):
    with gzip.open(freqs_output_loc, 'wt', encoding='utf-8') as f:
        for freq, token in freqs:
            f.write(f'{freq:>8} {token}\n')


def select_vectors(vectors_loc, candidate_tokens):
    wv = KeyedVectors.load_word2vec_format(vectors_loc, binary=True, unicode_errors='ignore')

    n = 0
    tokens_with_vector = []
    vectors = []
    for token in candidate_tokens:
        if token in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token])
        elif token.lower() in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token.lower()])
        elif (re.match(r'^[-‐][^\W\d_]+$', token) or re.match(r'^\+\d+$', token)) and token[1:] in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token[1:]])
        elif (re.match(r'^[-‐][^\W\d_]+$', token) or re.match(r'^\+\d+$', token)) and token[1:].lower() in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token[1:].lower()])
        elif re.match(r'^[^\W\d_]+[-‐]$', token) and token[:-1] in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token[:-1]])
        elif re.match(r'^[^\W\d_]+[-‐]$', token) and token[:-1].lower() in wv:
            tokens_with_vector.append(token)
            vectors.append(wv[token[:-1].lower()])
        else:
            print(f'Vector missing: {token}')
            n += 1

    print(f'Number of tokens without a vector: {n}')

    selected_wv = KeyedVectors(wv.vector_size)
    selected_wv.add_vectors(tokens_with_vector, vectors)
    return selected_wv


def is_valid_token(tokenizer, word):
    tokens = tokenizer(word)
    return bool(
        (len(word) < 40) and
        (len(tokens) == 1) and
        (word == '/' or '/' not in word or re.match(r'^[0-9]+/[0-9]+(/[0-9]+)?$', word)) and
        (not any(x in word for x in '\u0080\u0094\u0096\u0097\u200b\\~|=') or
         word in ['|', '~', '\\', '=']) and
        not re.match(r'^@\d+$', word) and
        not (word.startswith('.') and len(word) <= 3 and re.search(r'\d', word)) and
        # looks like an email adress
        not re.match(r'^[a-z0-9.]+@[a-z0-9.]+\.[a-z]+$', word, re.IGNORECASE))


if __name__ == '__main__':
    typer.run(main)
