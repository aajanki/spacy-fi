# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import NOUN, PROPN, PRON


def noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "nsubj",
        "nsubj:cop",
        "obj",
        "obl",
        "nmod",
        #"nmod:poss",
        "appos",
        "ROOT",
    ]
    labels_right = [
        "nmod",
        "nmod:poss",
    ]

    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings[label] for label in labels]
    np_deps_right = [doc.vocab.strings[label] for label in labels_right]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")

    rbracket = 0
    for i, word in enumerate(obj):
        if i < rbracket:
            continue
        if word.pos in (NOUN, PROPN, PRON) and word.dep in np_deps:
            rbracket = word.i + 1
            # try to extend the span to the right
            # to capture close apposition/possessive nominal constructions
            for rdep in doc[word.i].rights:
                if rdep.pos in (NOUN, PROPN) and rdep.dep in np_deps_right:
                    rbracket = rdep.i + 1
            yield word.left_edge.i, rbracket, np_label


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
