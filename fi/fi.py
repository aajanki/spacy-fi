from .lemmatizer import FinnishLemmatizer
from .syntax_iterators import SYNTAX_ITERATORS
from spacy.lang.fi import FinnishDefaults
from spacy.language import Language


class FinnishExDefaults(FinnishDefaults):
    syntax_iterators = SYNTAX_ITERATORS
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
