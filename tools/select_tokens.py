import gzip
import re
import plac
from pathlib import Path
from spacy.util import get_lang_class
from gensim.models import KeyedVectors
from tqdm import tqdm


@plac.annotations(
    freqs_loc=('Path to the input word frequencies', 'positional', None, Path),
    vectors_loc=('Path to the input word2vec binary file', 'positional', None, Path),
    freqs_output_loc=('Output path for the selected frequencies', 'positional', None, Path),
    vectors_output_loc=('Output path for the word2vec text file', 'positional', None, Path),
    num_tokens=('Maximum number of vecotrs to include', 'option', 'n', int)
)
def main(freqs_loc, vectors_loc, freqs_output_loc, vectors_output_loc, num_tokens=100000):
    print(f'Selecting {num_tokens} most frequent tokens')
    selected_tokens = select_tokens(freqs_loc, num_tokens)
    save_frequencies(freqs_output_loc, selected_tokens)

    print('Collecting vectors')
    output_wv = select_vectors(vectors_loc, selected_tokens)
    output_wv.save_word2vec_format(vectors_output_loc, binary=False)


def select_tokens(freqs_loc, num_tokens):
    tokenizer = get_lang_class("xx").Defaults.create_tokenizer()
    selected_tokens = []
    t = tqdm(total=num_tokens)
    freqs = (x.strip().split(' ', 1) for x in gzip.open(freqs_loc, 'rt', encoding='utf-8').readlines())
    for freq, token in freqs:
        if is_valid_token(tokenizer, token):
            selected_tokens.append((freq, token))
            t.update()

            if len(selected_tokens) >= num_tokens:
                break

    t.close()

    return selected_tokens


def save_frequencies(freqs_output_loc, freqs):
    with gzip.open(freqs_output_loc, 'wt', encoding='utf-8') as f:
        for freq, token in freqs:
            f.write(f'{freq:>8} {token}\n')


def select_vectors(vectors_loc, selected_tokens):
    wv = KeyedVectors.load_word2vec_format(vectors_loc, binary=True, unicode_errors='ignore')

    n = 0
    tokens = []
    vectors = []
    for _, token in tqdm(selected_tokens):
        if token in wv:
            tokens.append(token)
            vectors.append(wv[token])
        elif token.lower() in wv:
            tokens.append(token)
            vectors.append(wv[token.lower()])
        else:
            print(f'Vector missing: {token}')
            n += 1

    print(f'Number of tokens without a vector: {n}')

    selected_wv = KeyedVectors(wv.vector_size)
    selected_wv.add(tokens, vectors)
    return selected_wv


def is_valid_token(tokenizer, word):
    tokens = tokenizer(word)
    return ((len(word) > 1 or word.isdigit() or word in '.,:;?!()[]{}/\\"\'&%#<>|-_+=$€£@~*^') and len(word) < 40 and
            (len(tokens) == 1 or
             # special case for hyphenated words
             re.match(r'^[A-ZÅÄÖ0-9]+-[A-ZÅÄÖ0-9]+$', word, re.IGNORECASE)) and
            ('/' not in word or re.match(r'^[0-9]+/[0-9]+(/[0-9]+)?$', word)) and
            (not any(x in word for x in '\u0080\u0094\u0096\u0097\u200b\\~|=') or
             word in ['|', '~', '\\', '=']) and
            not re.match(r'^@\d+$', word) and
            not (word.startswith('.') and len(word) <= 3 and re.search(r'\d', word)) and
            # looks like an email adress
            not re.match(r'^[a-z0-9.]+@[a-z0-9.]+\.[a-z]+$', word, re.IGNORECASE))


if __name__ == '__main__':
    plac.call(main)
