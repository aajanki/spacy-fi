# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import NOUN, PROPN


def noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.

    Reference: http://scripta.kotus.fi/visk/sisallys.php?p=562
    """

    labels = [
        "acl",
        "advmod",
        "amod",
        "case",
        "det",
        "nmod",
        "nmod:poss",
        "nummod",
    ]

    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings[label] for label in labels]
    np_label = doc.vocab.strings.add("NP")

    def extract_nps(word):
        if word.pos in (NOUN, PROPN):
            left_i = word.i
            for t in word.lefts:
                if t.dep not in np_deps:
                    yield from extract_nps(t)
                else:
                    left_i = t.left_edge.i
                    break

            right_i = word.i
            if word.i < len(word.doc) - 1:
                t = word.nbor()
                while t.i <= word.right_edge.i and t.dep in np_deps:
                    right_i = t.i

                    if t.i >= len(word.doc) - 1:
                        break
                    
                    t = t.nbor()

            yield left_i, right_i + 1, np_label

            for i in range(right_i + 1, word.right_edge.i + 1):
                if word.doc[i].head.i <= right_i:
                    yield from extract_nps(word.doc[i])

        else:
            for t in word.children:
                yield from extract_nps(t)


    root = find_root(doc)
    yield from extract_nps(root)


def find_root(doc):
    root = doc.vocab.strings.add("ROOT")
    for word in doc:
        if word.dep == root:
            return word

    assert False, 'No root node!'


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
