"""A wrapper around spacy cli tools

This injects a tag_map to the Finnish language data and then forwards
the command line arguments to the spacy cli.
"""

import sys
import plac
from spacy import cli
from spacy import util
from spacy.symbols import POS, PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, X, VERB
from spacy.symbols import NOUN, PROPN, PART, INTJ, SPACE, PRON

TAG_MAP = {
    'Adv': {POS: ADV},
    'Pron': {POS: PRON},
    'V': {POS: VERB},
    'Punct': {POS: PUNCT},
    'N': {POS: NOUN},
    'A': {POS: ADJ},
    'Num': {POS: NUM},
    'C': {POS: CCONJ},
    'Adp': {POS: ADP},
    'Symb': {POS: SYM},
    'Interj': {POS: INTJ},
    'Foreign': {POS: X},
    'Adj': {POS: ADJ}
}


def main():
    fi = util.get_lang_class('fi')
    fi.Defaults.tag_map = TAG_MAP

    command_name = sys.argv[1].replace('-', '_')
    command = getattr(cli, command_name)
    plac.call(command, sys.argv[2:])


if __name__ == '__main__':
    main()
