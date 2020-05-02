# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import NOUN, PROPN, conj


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
    ]
    labels_internal = [
        "amod",
        "nmod",
        "nmod:poss",
        "nummod",
    ]
    labels_cop = [
        "cop",
        "nsubj:cop",
    ]

    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings[label] for label in labels]
    np_internal_deps = [doc.vocab.strings[label] for label in labels_internal]
    cop_deps = [doc.vocab.strings[label] for label in labels_cop]
    np_label = doc.vocab.strings.add("NP")
    root = doc.vocab.strings.add("ROOT")

    rbracket = 0
    for i, word in enumerate(obj):
        if i < rbracket:
            continue

        if word.pos in (NOUN, PROPN) and word.dep in np_deps:
            rbracket = extend_np_right(word, np_internal_deps)
            yield word.left_edge.i, rbracket, np_label
        elif word.pos in (NOUN, PROPN) and word.dep == root:
            left_i = word.left_edge.i
            for child in word.children:
                if child.dep in cop_deps:
                    left_i = child.right_edge.i + 1
            rbracket = word.right_edge.i + 1
            yield left_i, rbracket, np_label
        elif word.dep == conj and word.head.i < word.i:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head

            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.pos in (NOUN, PROPN) and head.dep in np_deps:
                left_i = shrink_np_left(word, np_internal_deps)
                rbracket = extend_np_right(word, np_internal_deps)
                yield left_i, rbracket, np_label


def shrink_np_left(word, valid_deps):
    # Skip punctuation and coordinating conjuction on the left side of
    # the word's subtree
    left = word.left_edge
    while left.dep not in valid_deps and left.i < word.i:
        left = left.nbor()

    return left.i


def extend_np_right(word, valid_deps):
    # try to extend the span to the right
    # to capture close apposition/possessive nominal constructions
    right_i = word.i + 1
    for rdep in word.rights:
        if rdep.pos == NOUN and rdep.dep in valid_deps:
            right_i = rdep.i + 1

    return right_i


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
