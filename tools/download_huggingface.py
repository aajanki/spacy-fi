import copy
import re
import typer
import unicodedata
from itertools import islice
from pathlib import Path
from datasets import load_dataset
from lingua import Language, LanguageDetectorBuilder
from more_itertools import ichunked
from tqdm import tqdm
from typing import Optional
from spacy.lang.char_classes import ALPHA
from .io import open_output
from urllib.parse import urlparse
from ftfy import fix_text
from .classifiers import SpamClassifier, CodeClassifier

skip_tlds = set([
    '.nl',
    '.be',
    '.one',
    '.pw',
])

time_inside_word_re = re.compile(r'(?<=[{a}]{{2}})(\d{{2}}[.:]\d{{2}})(?=[{a}])'.format(a=ALPHA))
word_and_url_re = re.compile(r'(?<=\w{3})https?://[-:/a-zA-Z0-9_.+%/?=#]+')

detect_languages = [
    Language.FINNISH,
    Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.ESTONIAN,
    Language.RUSSIAN,
    Language.SWEDISH
]

def main(
    dataset_name: str,
    subset: str,
    max_texts: int,
    batch_size: int,
    output_path: Path,
):
    # See the spam classifier training scripts at
    # https://github.com/aajanki/finnish-word-frequencies
    spam_classifier = SpamClassifier('tools/spammodels/spam_classifier_weights.json')
    code_classifier = CodeClassifier('tools/spammodels/code_classifier_weights.json')

    langdetector = LanguageDetectorBuilder.from_languages(*detect_languages).build()

    output_path.mkdir(exist_ok=True, parents=True)
    for p in output_path.glob('*.txt.bz2'):
        p.unlink()

    dataset = load_dataset(dataset_name, subset, split="train", streaming=True, trust_remote_code=True)
    dataset = (cleanup_text(x) for x in dataset)
    dataset = (
        x for x in dataset
        if not (is_spam(x, spam_classifier) or is_code(x, code_classifier))
    )
    texts = (x['text'] for x in dataset if is_body_text(x['text']))
    texts = (cleanup_punctuation(x) for x in texts)
    texts = (x for x in texts if is_finnish(langdetector, x))
    texts = islice(texts, max_texts)
    texts = tqdm(texts, total=max_texts, smoothing=0.02)

    for i, batch in enumerate(ichunked(texts, batch_size)):
        output_file = output_path / f'mc4_{i:02d}.txt.bz2'
        with open_output(output_file) as outf:
            for text in batch:
                outf.write(text)
                outf.write('\n')


def cleanup_text(x):
    text = x['text']
    text = unicodedata.normalize('NFC', text.strip())
    text = fix_text(text, uncurl_quotes=False)

    # Remove duplicated title
    # Many sites have the same text and the page title (first line) and the main
    # header (second line). Try to remove the duplication.
    #
    # Example:
    #
    # Ei suuruudenhullua, vaan reilua ja oikein | Näkökulma | Yleisradio | yle.fi
    # Ei suuruudenhullua, vaan reilua ja oikein
    lines = text.split('\n')
    if len(lines) > 1:
        title_re = re.compile(re.escape(lines[1]) + r' [-–|«]', re.IGNORECASE)
        remove_first_line = title_re.match(lines[0])

        if remove_first_line:
            text = '\n'.join(lines[1:])

    text = remove_bbcode(text)
    text = remove_sort_entry(text)

    # Cleanup "Riku Rantalahttp://www.hs.fi/haku/?query=riku+rantala"
    text = word_and_url_re.sub('', text)

    # Add space around a number in certain cases
    text = time_inside_word_re.sub(r' \1 ', text)

    out = copy.deepcopy(x)
    out['text'] = text

    return out


def is_body_text(text):
    # Skip lines where umlauts have been replaced by the Unicode
    # REPLACEMENT CHARACTER U+FFFD.
    if text.count('\ufffd') > 0:
        return False

    # Skip empty documents
    tokens = text.split()
    num_tokens = len(tokens)
    if num_tokens == 0:
        return False

    # Skip if digits occur too frequently.
    num_digit_tokens = sum(1 for t in tokens if re.match(r'^\(?[0-9][0-9.,:;]+\)?$', t))
    if num_digit_tokens / num_tokens > 0.33:
        return False

    # Skip if too many single characters tokens.
    num_single_character_tokens = sum(1 for t in tokens if len(t) == 1)
    if num_single_character_tokens / num_tokens > 0.25:
        return False

    # Very long words might indicate that this is programming language or data
    num_long_words = sum(
        1 for t in tokens
        if len(t) >= 40 and not t.startswith('http') and not t.startswith('www') and not ',' in t
    )
    if num_long_words / num_tokens > 0.01:
        return False

    # Skip page with lots of Wiki markup
    if maybe_wiki_markup(text):
        return False

    return True


# MC4 has already done language detection, but it's not perfect. Let's
# reduce false positives by doing a second pass of language detection
# with lingua.
#
# A document is considered Finnish if more than 80% of text is in Finnish.
# The threshold allows some embedded text on other languages. The threshold
# is selected to lean on excluding documents on unclear cases.
def is_finnish(langdetector, text):
    return finnish_proportion(langdetector, text) >= 0.8


def finnish_proportion(detector, text):
    languages = detector.detect_multiple_languages_of(text)
    fi_span_lengths = [
        lang.end_index - lang.start_index
        for lang in languages
        if lang.language == Language.FINNISH
    ]
    fi_total = sum(fi_span_lengths)

    return fi_total / len(text)


def cleanup_punctuation(text):
    text = re.sub(r'[\s\u2800]+', ' ', text)
    text = re.sub(r'[\u0000-\u0008\u000E-\u001F\u007F-\u0084\u0086-\u009F\u00AD\u200B-\u200D\u2060\uFEFF\uFFF0-\uFFFF]', '', text)
    text = re.sub(r'[\u0530-\u1DBF\u2C00-\uA6FF\uA800-\uAB2F\uAB70-\uD7FF\uE000-\uFAFF\uFB50-\uFDFF]+', ' ', text)
    text = re.sub(r'(?<=\s)[\ufe00-\ufe0f]+', '', text)
    text = re.sub(r'\s\.(?=[{a}]{{4}})'.format(a=ALPHA), '. ', text)
    text = re.sub(r'\.\.\.(?=[-+*/!?%(),:;<>€$£"\'])', '... ', text)
    text = re.sub(r'([.,|@~-])\1{9,}', r'\1\1\1\1\1\1\1\1\1\1', text)
    return text


def maybe_wiki_markup(text):
    """Heuristics for detecting wiki markup."""
    if len(text) == 0:
        return False

    num_link_boundaries = sum(1 for _ in re.finditer(r'\[\[|]]', text))
    if num_link_boundaries > 0:
        num_h2 = sum(1 for _ in re.finditer(r'==', text))
        num_pipes = text.count('|')
        wiki_token_freq = (num_link_boundaries + num_h2 + num_pipes) / len(text)
        return wiki_token_freq > 0.02
    else:
        return False


def remove_bbcode(text):
    """Remove the most common BBCode tags."""
    tags = r'|'.join([
        r'(?:\[URL="?[-a-zA-Z0-9.,:/$_@&+!*()%#]+"?])',
        r'(?:\[/URL])',
        r'(?:\[/?B])',
        r'(?:\[/?LEFT])',
    ])

    text = re.sub(tags, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[IMG][-a-zA-Z0-9.,:/$_@&+!*()%#]+\[/IMG]', ' ', text, flags=re.IGNORECASE)
    return text


def remove_sort_entry(text):
    """Remove SortEntry code block appearing in many e-commerce sites."""
    return re.sub(r'loc_fi_FI, sid_[A-Z0-9]+, prod, sort_\[SortEntry\(order=[A-Z_]+, direction=[A-Z_]+\)]', ' ', text)


def is_spam(x, spam_classifier):
    return (
        is_spam_url(x['url']) or
        spam_classifier.predict(x['text']) or
        'Hotels.comSee this page in English' in x['text']
    )


def is_code(x, code_classifier):
    return code_classifier.predict(x['text'])


def is_spam_url(url):
    dom = domain(url)
    tld = '.' + dom.split('.')[-1]
    return (dom == 'propilkki.ddns.net') or (tld in skip_tlds)


def domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    # Remove port
    netloc = netloc.split(':', 1)[0]
    return netloc


if __name__ == '__main__':
    typer.run(main)
