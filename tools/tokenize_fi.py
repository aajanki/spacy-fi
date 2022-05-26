import re
import typer
from pathlib import Path
from spacy.lang.fi import FinnishDefaults
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_ICONS
from spacy.lang.char_classes import LIST_HYPHENS, CURRENCY, UNITS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA, ALPHA_LOWER, ALPHA_UPPER, PUNCT
from spacy.language import Language
from tqdm import tqdm

TOKENIZER_PREFIXES = FinnishDefaults().prefixes + ['\\.']

QUOTES2 = CONCAT_QUOTES.replace("'", "")
DASHES = "".join(x for x in LIST_HYPHENS if len(x) == 1 and x != "-")
TOKENIZER_INFIXES = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[{al}])\.(?=[{au}0-9])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
        r"(?<=[{a}])[,!?{q}\)\]\(\[](?=[{a}])".format(a=ALPHA, q=QUOTES2),
        r"(?<=[{a}0-9])[!?<>=;](?=[{a}0-9])".format(a=ALPHA),
        r"(?<=[{a}])(?:[{d}/])(?=[{a}])".format(a=ALPHA, d=DASHES),
        r"(?<=[0-9])[-–+](?=[0-9])",
        r"(?<=[{a}])[()](?=[0-9])".format(a=ALPHA),
        r"(?<=[0-9])[()](?=[{a}])".format(a=ALPHA),
    ]
)

CURRENCY2 = CURRENCY + '|e'
TOKENIZER_SUFFIXES = (
    LIST_PUNCT
    + LIST_ELLIPSES
    + LIST_QUOTES
    + LIST_ICONS
    + ["—", "–"]
    + [
        r"(?<=[0-9])\+",
        r"(?<=°[FfCcKk])\.",
        r"(?<=[0-9])(?:[{u}{c}])".format(u=UNITS, c=CURRENCY2),
        r"(?<=[{al}{e}{p}{q}{c}])\.".format(
            al=ALPHA_LOWER, e=r"%²\-\+", q=CONCAT_QUOTES, c=CURRENCY2, p=PUNCT
        ),
        r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER),
        r"(?<=[^{a}0-9])-".format(a=ALPHA)
    ]
)

class FinnishCustomTokenizerDefaults(FinnishDefaults):
    prefixes = TOKENIZER_PREFIXES
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES

class FinnishCustomTokenizer(Language):
    lang = 'fi'
    Defaults = FinnishCustomTokenizerDefaults


def main(input_file: Path, output_file: Path):
    tokenizer = create_tokenizer()

    with input_file.open() as inf, output_file.open('w') as outf:
        for i, line in enumerate(tqdm(inf)):
            line = line.rstrip('\n')

            # MC4_fi_cleaned contains many cases where the full stop is missing
            # surrounding space. Adding a space helps spacy tokenizer parse
            # those better. (This might break up some smileys.)
            line = re.sub(r'(?<=[-,:;!?%()\[\]])([.,?!;])', r' \1', line)
            line = re.sub(r'([.,?!])(?=[-,:;!?%()\[\]])', r'\1 ', line)

            tokens = (x.text for x in tokenizer(line))
            outf.write(' '.join(tokens))
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
