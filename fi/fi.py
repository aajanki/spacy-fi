from .lemmatizer import FinnishLemmatizer
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_SUFFIXES
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from spacy.lang.fi import FinnishDefaults
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.symbols import POS, PUNCT, SYM, ADJ, CCONJ, SCONJ, NUM, DET, ADV
from spacy.symbols import ADP, X, VERB, NOUN, PROPN, PART, INTJ, SPACE, PRON, AUX
from spacy.util import update_exc


TAG_MAP = {
    'Adv': {POS: ADV},
    'Pron': {POS: PRON},
    'Propn': {POS: PROPN},
    'V': {POS: VERB},
    'Aux': {POS: AUX},
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
    tag_map = TAG_MAP
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES
    tokenizer_exceptions = update_exc(BASE_EXCEPTIONS, TOKENIZER_EXCEPTIONS)
    resources = [
        ('lemma_exc', 'lookups/fi_lemma_exc.json'),
    ]

    @classmethod
    def create_lemmatizer(cls, nlp=None, lookups=None):
        if lookups is None:
            lookups = cls.create_lookups(nlp=nlp)
        return FinnishLemmatizer(lookups)


class FinnishEx(Language):
    """This extends the SpaCy Finnish language class by adding more
    lexical properties.
    """

    lang = 'fi'
    Defaults = FinnishExDefaults
