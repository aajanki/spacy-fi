import pytest
from fi.fi import FinnishExtended

fi_nlp = FinnishExtended()
fi_tokenizer = fi_nlp.tokenizer

FI_TOKENIZER_TEST_EXAMPLES = [
    (
        'Nopeusrajoitus on 120km/h!',
        ['Nopeusrajoitus', 'on', '120', 'km/h', '!']
    ),
    (
        'Jäätelö maksaa 4,30e',
        ['Jäätelö', 'maksaa', '4,30', 'e']
    ),
    (
        'Hollanti sulkee hiilivoimalan.Parlamentti äänesti asian puolesta(selvin numeroin).',
        [
            'Hollanti', 'sulkee', 'hiilivoimalan', '.', 'Parlamentti', 'äänesti',
            'asian', 'puolesta', '(', 'selvin', 'numeroin', ')', '.'
        ]
    ),
    (
        'Tappelukerhon 1. sääntö: "Tappelukerhosta ei puhuta!"',
        [
            'Tappelukerhon', '1.', 'sääntö', ':', '"', 'Tappelukerhosta', 'ei',
            'puhuta', '!', '"'
        ]
    ),
    (
        'Noudettu kohteesta https://fi.wikipedia.org/w/index.php?title=Claudius',
        ['Noudettu', 'kohteesta', 'https://fi.wikipedia.org/w/index.php?title=Claudius']
    ),
    (
        'LOL :D =D :-) xD',
        ['LOL', ':D', '=D', ':-)', 'xD']
    ),
    (
        # TODO: should this be tokenized as '30.3.'?
        'Ministeri totesi(30.3.): "En kommentoi!"',
        [
            'Ministeri', 'totesi', '(', '30.3', '.', '):', '"', 'En',
            'kommentoi', '!', '"'
        ]
    ),
]


def tokenize(text):
    doc = fi_tokenizer(text)
    return [t.orth_ for t in doc]


@pytest.mark.parametrize(
    "text,expected_tokens", FI_TOKENIZER_TEST_EXAMPLES
)
def test_fi_tokenizer(text, expected_tokens):
    tokens = tokenize(text)

    assert tokens == expected_tokens
