from spacy.lang.fi import FinnishDefaults
from spacy.language import Language
from spacy.symbols import POS, PUNCT, SYM, ADJ, CCONJ, SCONJ, NUM, DET, ADV
from spacy.symbols import ADP, X, VERB, NOUN, PROPN, PART, INTJ, SPACE, PRON
from voikko_lemmatizer import VoikkoLemmatizer

TAG_MAP = {
    'Adv': {POS: ADV},
    'Pron': {POS: PRON},
    'Propn': {POS: PROPN},
    'V': {POS: VERB},
    'Punct': {POS: PUNCT},
    'N': {POS: NOUN},
    'A': {POS: ADJ},
    'Num': {POS: NUM},
    'C': {POS: CCONJ},
    'SC': {POS: SCONJ},
    'Adp': {POS: ADP},
    'Symb': {POS: SYM},
    'Interj': {POS: INTJ},
    'Foreign': {POS: X},
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
