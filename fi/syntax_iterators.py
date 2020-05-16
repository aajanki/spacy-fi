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
        "amod",
        "appos",
        "case",
        "compound",
        "compound:nn",
        "det",
        "flat:name",
        "nmod",
        "nmod:gobj",
        "nmod:gsubj",
        "nmod:poss",
        "nummod",
        "obj",
        "obl",
    ]

    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings[label] for label in labels]
    np_label = doc.vocab.strings.add("NP")

    def extract_nps(word):
        if word.pos in (NOUN, PROPN):
            left_i = word.i
            for t in reversed(list(word.lefts)):
                if t.dep in np_deps:
                    left_i = t.left_edge.i
                else:
                    break

            for t in word.lefts:
                if t.i < left_i:
                    yield from extract_nps(t)

            right_i = word.i
            for t in word.rights:
                if t.dep in np_deps:
                    right_i = t.right_edge.i
                else:
                    break

            yield left_i, right_i + 1, np_label

            for i in range(right_i + 1, word.right_edge.i + 1):
                candidate_word = word.doc[i]
                if candidate_word.head.i <= right_i:
                    yield from extract_nps(candidate_word)

        else:
            for t in word.children:
                yield from extract_nps(t)


    root = find_root(obj)
    yield from extract_nps(root)


def find_root(doc):
    root = doc.vocab.strings.add("ROOT")
    for word in doc:
        if word.dep == root:
            return word

    assert False, 'No root node!'


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
