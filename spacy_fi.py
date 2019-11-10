"""A wrapper around spacy cli tools

This injects some more lexical properties to the Finnish language
class and then forwards the command line arguments to the spacy cli.
"""

import sys
import plac
from spacy import cli, util
from spacy.lang.fi import FinnishDefaults
from spacy.language import Language
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


class FinnishExDefaults(FinnishDefaults):
    @classmethod
    def create_lookups(cls, nlp=None):
        lemma_lookup = util.load_language_data('data/fi_lemma_lookup.json')
        lookups = FinnishDefaults.create_lookups()
        lookups.add_table('lemma_lookup', lemma_lookup)
        return lookups

    tag_map = TAG_MAP


class FinnishEx(Language):
    lang = 'fi'
    Defaults = FinnishExDefaults


if __name__ == '__main__':
    util.set_lang_class('fi', FinnishEx)

    command_name = sys.argv[1].replace('-', '_')
    command = getattr(cli, command_name)
    plac.call(command, sys.argv[2:])
