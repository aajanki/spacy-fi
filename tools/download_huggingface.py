import langid
import re
import typer
import unicodedata
from itertools import islice
from pathlib import Path
from datasets import load_dataset
from more_itertools import ichunked
from tqdm import tqdm
from typing import Optional
from spacy.lang.char_classes import ALPHA
from .io import open_output
from urllib.parse import urlparse

spam_words = [
    'seksitreffit', 'sextreffit', 'seksiseuraa', 'seksideitti', 'sex',
    'suomiporno', 'suomipornovideot', 'seksivideot', 'seksiväline',
    'cumshot', 'escort', 'webcam',
    'kasino', 'kasinot', r'netti[ck]asino\w*?', 'poker', 'slot',
    'casino', 'slots', 'blackjack', 'roulette', 'holdem', 'lotto',
    r'spelautomat\w*', 'spela', 'gratis',
    'the hotel', 'bed', 'resort', 'inn', 'waterfront', 'bonus',
    r'alennuskood\w+?', r'alekood\w+?'
    'lahjakortit', 'tarjouskoodi', 'hinta', 'free', 'price', 'discount',
    'pikavippi', 'pikavipit', 'pikavippiä', r'pikalain\w+?', 'luottopäätös',
    'vakuuksia', 'prescriptions?', 'citrate', 'adderall', 'side effects',
    'dosage', 'recreational', 'ilman reseptiä', 'reseptivapaa', 'substitute',
    'pharmacy', 'generic', 'apteekki', 'apteekissa', 'pills',
     r'\w{35}\w*?(?=\s)',
]
really_spammy_words = [
    'sildenafil', 'sildenafiili', 'tadalafil', 'provigil', 'modafinil',
    'modalert', 'nuvigil', 'viagra', 'cialis', 'subutex', 'erotic',
    'massage', 'milfs?',
]
skip_domains = set([
    'www.spreadshirt.fi', # word salad
    'www.fi.freelancer.com', # a lot of non-Finnish content
    'fi.hotels.com', # very long words, non-Finnish
    'ssl-fi.hotels.com', # very long words, non-Finnish
    'et.hotels.com', # very long words, non-Finnish
    'nl.hotels.com', # very long words, non-Finnish
    'sv.hotels.com', # very long words, non-Finnish
    'www.hotels.com', # very long words, non-Finnish
    'hintaseuranta.fi', # word salad
    'www.karkkainen.com', # JSON
    'www.dx.com', # word salad
    'www.fi.kayak.com', # lots of non-Finnish content
    'www.auto1.fi', # word salad
    'www.prisma.fi', # JSON
    'www.iltapulu.fi', # listing, very long words
    'tietokonekauppa.fi', # very long words
    'www.booking.com', # non-Finnish
    'www.zoover.fi', # repeating text
    'propilkki.ddns.net', # listing
    'www.vertaa.fi', # word salad
    'www.tripadvisor.fi', # JSON
    'www.airbnb.fi', # non-Finnish
    'www.opensubtitles.org', # listing, very long words
    'www.sokos.fi', # JSON
    'www.lightinthebox.com', # word salad
    'www.skruvat.fi', # word salad
    'www.shutterstock.com', # repeating text
    'www.dwensa.info', # listing
    'www.foreca.fi', # listing
    'www.suomalainen.com', # JSON
    'www.def-shop.fi', # listing
    'www.thomann.de', # non-Finnish, incorrect encoding, tags
    'www.fragrancenet.com', # word salad
    'www.on24.fi', # word salad, long words
    'tipsed.com', # word salad
    'www.gigantti.fi', # JSON
    'www.klingel.fi', # JSON
    'www.kodinterra.fi', # JSON
])
skip_tlds = set([
    '.nl',
    '.be',
])
spam_re = re.compile(
    '|'.join(r'\b' + w + r'\b' for w in spam_words),
    re.IGNORECASE
)
spam2_re = re.compile(
    '|'.join(r'\b' + w + r'\b' for w in really_spammy_words),
    re.IGNORECASE
)

time_inside_word_re = re.compile(r'(?<=[{a}]{{2}})(\d{{2}}[.:]\d{{2}})(?=[{a}])'.format(a=ALPHA))
word_and_url_re = re.compile(r'(?<=\w{3})https?://[-:/a-zA-Z0-9_.+%/?=#]+')


def main(
    dataset_name: str,
    subset: str,
    max_texts: int,
    batch_size: int,
    output_path: Path,
    cleanup: Optional[bool] = None,
):
    output_path.mkdir(exist_ok=True, parents=True)
    for p in output_path.glob('*.txt.bz2'):
        p.unlink()

    dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
    selected_lines = lines(dataset)
    if cleanup:
        selected_lines = (x for x in selected_lines if not is_spam_url(x['url']))
    texts = (x['text'] for x in selected_lines)
    if cleanup:
        texts = (cleanup_text(x) for x in texts)
        texts = (x for x in texts if is_clean_finnish(x))
    texts = (cleanup_punctuation(x) for x in texts)
    texts = islice(texts, max_texts)
    texts = tqdm(texts, total=max_texts, smoothing=0.05)

    for i, batch in enumerate(ichunked(texts, batch_size)):
        output_file = output_path / f'mc4_{i:02d}.txt.bz2'
        with open_output(output_file) as outf:
            for text in batch:
                outf.write(text)
                outf.write('\n')


def lines(dataset):
    for x in iter(dataset):
        text = unicodedata.normalize('NFC', x['text'].strip())
        if text:
            yield {'text': text, 'url': x['url']}


def is_clean_finnish(text):
    # We skip lines where umlauts have been replaced by the Unicode
    # REPLACEMENT CHARACTER U+FFFD.
    if text.count('\ufffd') > 0:
        return False

    # Skip if digits occur too frequently.
    tokens = text.split()
    num_tokens = len(tokens)
    if num_tokens == 0:
        return False

    num_digit_tokens = sum(1 for t in tokens if re.match(r'^\(?[0-9][0-9.,:;]+\)?$', t))
    if num_digit_tokens / num_tokens > 0.25:
        return False

    # Skip if too many single characters tokens.
    num_single_character_tokens = sum(1 for t in tokens if len(t) == 1)
    if num_single_character_tokens / num_tokens > 0.25:
        return False

    num_braces = sum(1 for t in tokens if t == '{' or t == '}')
    if num_braces / num_tokens > 0.05:
        return False

    # Very long words might indicate that this is some kind of programming language
    num_long_words = sum(1 for t in tokens if len(t) >= 50 and not t.startswith('http'))
    if num_long_words / num_tokens > 0.05:
        return False

    # Simple spam filter
    # Spam = pages that don't contain proper sentences (lists, machine
    # generated fake Finnish, etc.) or long repeated text segments.
    spam_score = sum(1 for _ in spam_re.finditer(text)) + \
                 4 * sum(1 for _ in spam2_re.finditer(text))
    if spam_score / num_tokens > 0.05:
        return False

    # Skip page with lots of Wiki markup
    if maybe_wiki_markup(text):
        return False

    # MC4 has already done language detection, but it's not perfect. Let's
    # reduce false positives but doing a second pass of language detection
    # with langid.
    #
    # We don't care even if some small fraction of lines gets skipped even
    # though they really are in Finnish. Since we are anyway taking only a
    # portion of lines, skipping some lines unnecessarily doesn't matter.
    #
    # The langid filtering has the additional benefit that it discards lines
    # with incorrect encodings.
    langcode, _ = langid.classify(text)
    return langcode == 'fi'


def cleanup_punctuation(text):
    text = re.sub(r'[\s\u2800]+', ' ', text)
    text = re.sub(r'[\u0000-\u0008\u000E-\u001F\u007F-\u0084\u0086-\u009F\u00AD\u200B-\u200D\u2060\uFEFF\uFFF0-\uFFFF]', '', text)
    text = re.sub(r'[\u0530-\u1DBF\u2C00-\uA6FF\uA800-\uAB2F\uAB70-\uD7FF\uE000-\uFAFF\uFB50-\uFDFF]+', ' ', text)
    text = re.sub(r'(?<=\s)[\ufe00-\ufe0f]+', '', text)
    text = re.sub(r'\s\.(?=[{a}]{{4}})'.format(a=ALPHA), '. ', text)
    text = re.sub(r'\.\.\.(?=[-+*/!?%(),:;<>€$£"\'])', '... ', text)
    return text


def cleanup_text(text):
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
        lines_to_remove = []
        title_re = re.compile(re.escape(lines[1]) + r' [-–|«]', re.IGNORECASE)
        if title_re.match(lines[0]):
            lines_to_remove.append(0)

        # Remove Javascript and JSON
        for i, line in enumerate(lines):
            if maybe_js_or_json(line):
                lines_to_remove.append(i)

        if lines_to_remove:
            lines = [lines[i] for i in range(len(lines)) if i not in lines_to_remove]
            text = '\n'.join(lines)

    text = remove_bbcode(text)
    text = remove_sort_entry(text)

    # Cleanup "Riku Rantalahttp://www.hs.fi/haku/?query=riku+rantala"
    text = word_and_url_re.sub('', text)

    # Double encoded UTF-8 umlauts
    text = text.replace('\u00C3\u00A4', 'ä') \
        .replace('\u00C3\u00A5', 'å') \
        .replace('\u00C3\u00B6', 'ö') \
        .replace('\u00C3\u0084', 'Ä') \
        .replace('\u00C3\u0085', 'Å') \
        .replace('\u00C3\u0096', 'Ö')

    # Add space around a number in certain cases
    text = time_inside_word_re.sub(r' \1 ', text)

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


def maybe_js_or_json(text):
    """Heuristics for detecting JSON or JavaScript."""

    # Possible false positives: "undefined" and "function" might occur in
    # English text and "var" might occur in Swedish text. Those do not
    # matter much because we are anyway interested in keeping Finnish text only.
    #
    # Sometimes in MC4 a line consists of a text title followed by javascript
    # code. This will incorrectly detect the whole line as javascript.
    #
    # FIXME: proper JavaScript detection
    common_js_keywords1 = [
        r'\bvar\s', r'\bnull\b', r'\bundefined\b', r'\bfunction\b',
        r'\b\|\|\b', r'\b\(\)\b', r'\b={2,3}\b',
    ]
    common_js_keywords2 = [r'{', r'}', r'\[', r']']
    js_re1 = re.compile('|'.join(common_js_keywords1), re.IGNORECASE)
    js_re2 = re.compile('|'.join(common_js_keywords2))

    num_js_tokens1 = sum(1 for _ in js_re1.finditer(text))
    num_js_tokens2 = sum(1 for _ in js_re2.finditer(text))
    num_js_tokens3 = text.count(';')
    looks_like_js = num_js_tokens1 >= 1 and \
                    num_js_tokens2 >= 2 and \
                    (num_js_tokens3 >= 1 or num_js_tokens2 >= 10)
    looks_like_json = re.match(r'\[\s*\{', text) is not None and \
                      re.search(r'}\s*]$', text) is not None

    return looks_like_js or looks_like_json


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


def is_spam_url(url):
    dom = domain(url)
    tld = '.' + dom.split('.')[-1]
    skip1 = dom in skip_domains
    skip2 = tld in skip_tlds
    return skip1 or skip2


def domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    # Remove port
    netloc = netloc.split(':', 1)[0]
    return netloc


if __name__ == '__main__':
    typer.run(main)
