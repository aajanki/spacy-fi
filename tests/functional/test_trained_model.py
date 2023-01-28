# This will run simple checks on all capabilities of a trained full model.
# The model is expected to be available at training/merged.

import pytest
import spacy
import fi

# Reload the model before each test to clear all state between tests
@pytest.fixture
def nlp():
    return spacy.load('training/merged')


POS_TEST_CASES = [
    (
        'Hämeen linna sijaitsee Vanajaveden rannalla Hämeenlinnassa.',
        ['PROPN', 'NOUN', 'VERB', 'PROPN', 'NOUN', 'PROPN', 'PUNCT']
    ),
    (
        'Maitohampaiden puhjettua lapselle hankitaan sopiva hammasharja.',
        ['NOUN', 'VERB', 'NOUN', 'VERB', 'ADJ', 'NOUN', 'PUNCT']
    ),
]

@pytest.mark.parametrize('text,expected_pos', POS_TEST_CASES)
def test_pos(nlp, text, expected_pos):
    doc = nlp(text)
    observed_pos = [t.pos_ for t in doc]

    assert observed_pos == expected_pos


DEP_TEST_CASES = [
    (
        'Mieltä kiehtoo ajatus tähtitieteen opiskelusta yliopistossa.',
        ['kiehtoo', 'kiehtoo', 'kiehtoo', 'opiskelusta', 'ajatus', 'opiskelusta', 'kiehtoo']
    ),
]

@pytest.mark.parametrize('text,expected_head', DEP_TEST_CASES)
def test_dep(nlp, text, expected_head):
    doc = nlp(text)
    observed_head = [t.head.text for t in doc]

    assert observed_head == expected_head


LEMMA_TEST_CASES = [
    (
        'meidän asemassamme voisimme itse päättää aikatauluistamme.',
        ['minä', 'asema', 'voida', 'itse', 'päättää', 'aikataulu', '.']
    ),
    (
        'Mielestäni luontokadon kiihtyminen hidastuu liian hitaasti',
        ['mieli', 'luontokato', 'kiihtyminen', 'hidastua', 'liian', 'hitaasti']
    ),
]

@pytest.mark.parametrize('text,expected_lemma', LEMMA_TEST_CASES)
def test_lemma(nlp, text, expected_lemma):
    doc = nlp(text)
    observed_lemma = [t.lemma_ for t in doc]

    assert observed_lemma == expected_lemma


NER_TEST_CASES= [
    (
        'Mervi ja Matleena matkustavat Kööpenhaminaan perjantaina',
        ['B-PERSON', 'O', 'B-PERSON', 'O', 'B-GPE', 'B-DATE']
    ),
]

@pytest.mark.parametrize('text,expected_ner', NER_TEST_CASES)
def test_ner(nlp, text, expected_ner):
    doc = nlp(text)
    observed_ner = [
        f'{t.ent_iob_}{"-" + t.ent_type_ if t.ent_type_ else ""}' for t in doc
    ]

    assert observed_ner == expected_ner


NOUN_CHUNK_TEST_CASES = [
    (
        'Kaksi tyttöä potkii punaista palloa',
        ['Kaksi tyttöä', 'punaista palloa']
    ),
    (
        'Päivän kohokohta oli vierailu museossa kummilasten kanssa',
        ['kohokohta', 'vierailu', 'kummilasten'],
    ),
]

@pytest.mark.parametrize('text,expected_np_fragments', NOUN_CHUNK_TEST_CASES)
def test_noun_chunks(nlp, text, expected_np_fragments):
    doc = nlp(text)
    observed_np = [x.text for x in doc.noun_chunks]

    for np in expected_np_fragments:
        assert any(np in x for x in observed_np)


def test_morph(nlp):
    doc = nlp('Koulussa opittiin lukemaan')

    assert doc[0].morph.get('Case') == ['Ine']

    assert doc[1].morph.get('Voice') == ['Pass']
    assert doc[1].morph.get('Tense') == ['Past']

    assert doc[2].morph.get('VerbForm') == ['Inf']


def test_vectors(nlp):
    assert nlp.vocab['pölkky'].has_vector
    assert nlp.vocab['kiikareilla'].has_vector

    sim1 = nlp.vocab['kuningas'].similarity(nlp.vocab['kuningatar'])
    sim2 = nlp.vocab['kuningas'].similarity(nlp.vocab['autokorjaamo'])

    assert sim1 > sim2


def test_lexeme_prob(nlp):
    assert -7.0 < nlp.vocab['se'].prob < -5.0
    assert -9.0 < nlp.vocab['yksi'].prob < -6.0
    assert -18.0 < nlp.vocab['harmaapäätikka'].prob < -10.0
    assert nlp.vocab['lapset'].prob > nlp.vocab['läsnäoloa'].prob
    assert nlp.vocab['lentokone'].prob > nlp.vocab['höyrylaivalla'].prob
    assert nlp.vocab['minä'].prob > nlp.vocab['Heille'].prob
