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
from spacy.lang.char_classes import ALPHA, ALPHA_UPPER
from .io import open_output

spam_words = [
    'seksitreffit', 'sextreffit', 'seksiseuraa', 'seksideitti', 'sex',
    'suomiporno', 'suomipornovideot', 'seksivideot', 'seksiväline',
    'webcam',
    'kasino', 'kasinot', 'nettikasino', 'nettikasinot', 'poker',
    'casino', 'slots', 'blackjack', 'roulette', 'holdem',
    'hotelli', 'hotellit', 'Hotellitarjoukset', 'hotelliKirjaudu',
    '\w+?Hotellit', 'Hotels\.comSee',
    'Rewards', 'RewardsSaat', 'RewardsTietoa', 'palkintoyö\w*?',
    'sivuillammeMatkanvälittäjätTiedotusEhdot', '\)Hotels\.com™',
    'evästeistäAsiakaspalveluVarauksesiOhjeetPalautetta',
    'Hintaseuranta', 'alennuskood\w+?', 'alekood\w+?'
    'lahjakortit', 'tarjouskoodi', 'hinta', 'free', 'price', 'discount',
    'pikavippi', 'pikavipit', 'pikavippiä', 'pikalain\w+?', 'luottopäätös',
    'vakuuksia',
]
spam_re = re.compile(
    '|'.join(r'\b' + w + r'\b' for w in spam_words),
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
        selected_lines = (cleanup_text(x) for x in selected_lines)
        selected_lines = (x for x in selected_lines if is_clean_finnish(x))
    selected_lines = (cleanup_punctuation(x) for x in selected_lines)
    selected_lines = islice(selected_lines, max_texts)
    selected_lines = tqdm(selected_lines, total=max_texts, smoothing=0.05)

    for i, batch in enumerate(ichunked(selected_lines, batch_size)):
        output_file = output_path / f'mc4_{i:02d}.txt.bz2'
        with open_output(output_file) as outf:
            for text in batch:
                outf.write(text)
                outf.write('\n')


def lines(dataset):
    for x in iter(dataset):
        text = unicodedata.normalize('NFC', x['text'].strip())
        if text:
            yield text


def is_clean_finnish(text):
    # We skip lines where umlauts have been replaced by the Unicode
    # REPLACEMENT CHARACTER U+FFFD.
    if text.count('\ufffd') > 0:
        return False

    # Skip if digits occur too frequently.
    tokens = text.split()
    num_tokens = len(tokens)
    num_digit_tokens = sum(1 for t in tokens if re.match(r'^\(?[0-9][0-9.:]+\)?$', t))
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
    num_spam_tokens = sum(1 for _ in spam_re.finditer(text))
    if num_spam_tokens / num_tokens > 0.05:
        return False

    if 'auton...ostostavaihdostakoeajostalisätiedoistaVaihdossa' in tokens:
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
        title_re = re.compile(re.escape(lines[1]) + r' [-–|«]', re.IGNORECASE)
        if title_re.match(lines[0]):
            text = '\n'.join(lines[1:])

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


if __name__ == '__main__':
    typer.run(main)
