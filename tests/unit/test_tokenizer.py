import pytest
from textwrap import dedent
from fi.fi import FinnishExtended

fi_nlp = FinnishExtended()
fi_tokenizer = fi_nlp.tokenizer

FI_TOKENIZER_TEST_EXAMPLES = [
    (
        'Nopeusrajoitus on 120km/h!',
        ['Nopeusrajoitus', 'on', '120', 'km/h', '!']
    ),
    (
        'Jäätelö maksaa 4,30e ja kahvi 2,70€',
        ['Jäätelö', 'maksaa', '4,30', 'e', 'ja', 'kahvi', '2,70', '€']
    ),
    (
        'Hollanti sulkee hiilivoimalan.Parlamentti äänesti asian puolesta(selvin numeroin).',
        [
            'Hollanti', 'sulkee', 'hiilivoimalan', '.', 'Parlamentti', 'äänesti',
            'asian', 'puolesta', '(', 'selvin', 'numeroin', ')', '.'
        ]
    ),
    (
        'maailman 1. naispääministeri aloitti vuonna 1960.',
        ['maailman', '1.', 'naispääministeri', 'aloitti', 'vuonna', '1960', '.'],
    ),
    (
        'Tappelukerhon 1. sääntö: "Tappelukerhosta ei puhuta!"',
        [
            'Tappelukerhon', '1.', 'sääntö', ':', '"', 'Tappelukerhosta', 'ei',
            'puhuta', '!', '"'
        ]
    ),
    (
        'pelikoneeni on Xbox360 ja autoni Audi A5.',
        ['pelikoneeni', 'on', 'Xbox360', 'ja', 'autoni', 'Audi', 'A5', '.']
    ),
    (
        'Ministeri totesi(yrmeästi): "En kommentoi!"',
        [
            'Ministeri', 'totesi', '(', 'yrmeästi', '):', '"', 'En',
            'kommentoi', '!', '"'
        ]
    ),
    (
        'Noudettu kohteesta https://fi.wikipedia.org/w/index.php?title=Claudius',
        ['Noudettu', 'kohteesta', 'https://fi.wikipedia.org/w/index.php?title=Claudius']
    ),
    (
        'vasta-aine em—dash 1-3 4—7 8−9',
        ['vasta-aine', 'em', '—', 'dash', '1-3', '4—7', '8−9']
    ),
    (
        '2 euroa /hlö',
        ['2', 'euroa', '/', 'hlö']
    ),
    (
        dedent('''\
        •ensimmäinen
        ›toinen
        –kolmas
        →neljäs
        *viides'''),
        [
            '•', 'ensimmäinen', '\n', '›', 'toinen', '\n', '–', 'kolmas', '\n',
            '→', 'neljäs', '\n', '*', 'viides'
        ]
    ),
    (
        'LOL :D =D :-) (>_<) xD',
        ['LOL', ':D', '=D', ':-)', '(>_<)', 'xD']
    ),
]

FI_TOKENIZER_XFAIL_EXAMPLES = [
    (
        '\\web\\ /osoite/ https://fi.wikipedia.org/ ',
        ['\\', 'web', '\\', '/', 'osoite', '/', 'https://fi.wikipedia.org/']
    ),
    (
        'Tänään 9.4. juhlitaan suomen kielen päivää. 24.12. on jouluaatto',
        [
            'Tänään', '9.4.', 'juhlitaan', 'suomen', 'kielen', 'päivää', '.',
            '24.12.' 'on', 'jouluaatto'
        ]
    ),
    (
        'sivu 3(7) viite[5]',
        ['sivu', '3', '(', '7', ')', 'viite', '[', '5', ']']
    ),
    (
        '18+ 4-',
        ['18', '+', '4', '-']
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


@pytest.mark.parametrize(
    "text,expected_tokens", FI_TOKENIZER_XFAIL_EXAMPLES
)
@pytest.mark.xfail
def test_fi_tokenizer_xfail(text, expected_tokens):
    tokens = tokenize(text)

    assert tokens == expected_tokens
