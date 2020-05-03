# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import NOUN, PROPN, conj


def noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.

    Reference: http://scripta.kotus.fi/visk/sisallys.php?p=562
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
        "case",
    ]
    labels_cop = [
        "cop",
        "nsubj:cop",
    ]
    labels_root_or_nsubjcop = [
        "ROOT",
        "nsubj:cop",
    ]

    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings[label] for label in labels]
    np_internal_deps = [doc.vocab.strings[label] for label in labels_internal]
    cop_deps = [doc.vocab.strings[label] for label in labels_cop]
    deps_root_or_nsubjcop = [doc.vocab.strings[label] for label in labels_root_or_nsubjcop]
    np_label = doc.vocab.strings.add("NP")
    root = doc.vocab.strings.add("ROOT")

    rbracket = 0
    for i, word in enumerate(obj):
        if i < rbracket:
            continue

        if (word.pos in (NOUN, PROPN) and
            not has_noun_parent_non_copula(word, deps_root_or_nsubjcop)
            and word.dep in np_deps
        ):
            rbracket = shrink_np_right(word, np_internal_deps)
            yield word.left_edge.i, rbracket, np_label
        elif word.pos in (NOUN, PROPN) and word.dep == root:
            left_i = word.left_edge.i
            for child in word.children:
                if child.dep in cop_deps:
                    left_i = child.right_edge.i + 1
            rbracket = shrink_np_right(word, np_internal_deps)
            yield left_i, rbracket, np_label
        elif word.dep == conj and word.head.i < word.i:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head

            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.pos in (NOUN, PROPN) and head.dep in np_deps:
                left_i = shrink_np_left(word, np_internal_deps)
                rbracket = shrink_np_right(word, np_internal_deps)
                yield left_i, rbracket, np_label


def has_noun_parent_non_copula(word, invalid_dep):
    # For example in "Päivän kohokohta oli vierailu museossa", returns
    # True for "päivän" (has a parent noun "kohokohta") and False for
    # "kohokohta" (the parent noun "vierailu" is nsubj:cop).
    t = word
    while t.dep not in invalid_dep:
        t = t.head
        if t.pos in (NOUN, PROPN):
            return True

    return False


def shrink_np_left(word, valid_deps):
    # Skip punctuation and coordinating conjuction on the left side of
    # the word's subtree
    left = word.left_edge
    while left.dep not in valid_deps and left.i < word.i:
        left = left.nbor()

    return left.i


def shrink_np_right(word, valid_deps):
    for t in word.rights:
        if t.dep not in valid_deps:
            return t.left_edge.i

    return word.right_edge.i + 1


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
