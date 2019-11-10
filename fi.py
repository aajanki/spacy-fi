from spacy.lang.fi import FinnishDefaults
from spacy.language import Language
from spacy.symbols import POS, PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, X, VERB
from spacy.symbols import NOUN, PROPN, PART, INTJ, SPACE, PRON
from voikko_lemmatizer import VoikkoLemmatizer

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
    def create_lemmatizer(cls, nlp=None, lookups=None):
        return VoikkoLemmatizer(lookups)

    tag_map = TAG_MAP


class FinnishEx(Language):
    """This extends the SpaCy Finnish language class by adding more
    lexical properties.
    """

    lang = 'fi'
    Defaults = FinnishExDefaults
