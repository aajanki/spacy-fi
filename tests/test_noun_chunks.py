# coding: utf-8
from __future__ import unicode_literals

import pytest
import spacy.util
from fi.fi import FinnishEx
from util import get_doc_from_text

spacy.util.set_lang_class('fi', FinnishEx)
fi_tokenizer = FinnishEx.Defaults.create_tokenizer()


FI_NP_TEST_EXAMPLES = [
    (
        'Iloinen tyttö potkaisee punaista palloa',
        ['ADJ', 'NOUN', 'VERB', 'ADJ', 'NOUN'],
        ['amod', 'nsubj', 'ROOT', 'amod', 'obj'],
        [1, 1, 0, 1, -2],
        ['Iloinen tyttö', 'punaista palloa'],
    ),
    (
        'Erittäin vaarallinen leijona karkasi kiertävän sirkuksen eläintenkesyttäjältä',
        ['ADV', 'ADJ', 'NOUN', 'VERB', 'ADJ', 'NOUN', 'NOUN'],
        ['advmod', 'amod', 'nsubj', 'ROOT', 'amod', 'nmod:poss', 'obl'],
        [1, 1, 1, 0, 1, 1, -3],
        ['Erittäin vaarallinen leijona', 'kiertävän sirkuksen eläintenkesyttäjältä'],
    ),
    (
        'Leijona raidallisine tassuineen piileksii Porin kaupungin lähellä',
        ['NOUN', 'ADJ', 'NOUN', 'VERB', 'PROPN', 'NOUN', 'ADP'],
        ['nsubj', 'amod', 'nmod', 'ROOT', 'nmod:poss', 'obl', 'case'],
        [3, 1, -2, 0, 1, -2, -1],
        ['Leijona raidallisine tassuineen', 'Porin kaupungin'],
    ),
]


@pytest.mark.parametrize(
    "text,pos,deps,heads,expected_noun_chunks", FI_NP_TEST_EXAMPLES
)
def test_fi_noun_chunks(text, pos, deps, heads, expected_noun_chunks):
    doc = get_doc_from_text(text, fi_tokenizer, pos=pos, heads=heads, deps=deps)
    noun_chunks = list(doc.noun_chunks)

    assert len(noun_chunks) == len(expected_noun_chunks)
    for i, np in enumerate(noun_chunks):
        assert np.text == expected_noun_chunks[i]
