import sys
import typer
from pathlib import Path
from spacy.lang.fi import FinnishDefaults
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_ICONS
from spacy.lang.char_classes import LIST_HYPHENS, CURRENCY, UNITS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA, ALPHA_LOWER, ALPHA_UPPER, PUNCT
from spacy.language import Language
from tqdm import tqdm

QUOTES2 = CONCAT_QUOTES.replace("'", "")
DASHES = "|".join(x for x in LIST_HYPHENS if len(x) == 1 and x != "-")
TOKENIZER_INFIXES = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
        r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q=QUOTES2),
        r"(?<=[{a}])(?:{d})(?=[{a}])".format(a=ALPHA, d=DASHES),
        r"(?<=[{a}0-9])[<>=/](?=[{a}])".format(a=ALPHA),
        r"(?<=[0-9])[-–+](?=[0-9])",
    ]
)
TOKENIZER_SUFFIXES = (
    LIST_PUNCT
    + LIST_ELLIPSES
    + LIST_QUOTES
    + LIST_ICONS
    + ["—", "–"]
    + [
        r"(?<=[0-9])\+",
        r"(?<=°[FfCcKk])\.",
        r"(?<=[0-9])(?:{c})".format(c=CURRENCY),
        r"(?<=[0-9])(?:{u})".format(u=UNITS),
        r"(?<=[{al}{e}{p}(?:{q})])\.".format(
            al=ALPHA_LOWER, e=r"%²\-\+", q=CONCAT_QUOTES, p=PUNCT
        ),
        r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER),
    ]
)

class FinnishCustomTokenizerDefaults(FinnishDefaults):
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES

class FinnishCustomTokenizer(Language):
    lang = 'fi'
    Defaults = FinnishCustomTokenizerDefaults


def main(input_file: Path, output_file: Path):
    tokenizer = create_tokenizer()

    with input_file.open() as inf, output_file.open('w') as outf:
        for i, line in enumerate(tqdm(inf)):
            tokens = (x.text for x in tokenizer(line.rstrip('\n')))
            cleaned_tokens = (t.lstrip('.') if len(t) > 1 else t for t in tokens)
            outf.write(' '.join(cleaned_tokens))
            outf.write('\n')

            if i % 200000 == 0:
                # This will leak memory unless the tokenizer is re-created
                # periodically
                del tokenizer
                tokenizer = create_tokenizer()


def create_tokenizer():
    return FinnishCustomTokenizer().tokenizer


if __name__ == '__main__':
    typer.run(main)
