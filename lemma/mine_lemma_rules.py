import json
import re
import sys
import plac
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from collections import Counter, OrderedDict
from itertools import takewhile
from pathlib import Path
from voikko import libvoikko

pure_num_re = re.compile(r'^-?=q+(:p+)$')
parenthesis_re = re.compile(r'\((.+?)\)')


@plac.annotations(
    min_freq=('Minimum rule frequency', 'option', 'f', int),
    output=('Output path', 'option', None, Path)
)
def main(min_freq=10000, output=Path('data/lemma')):
    v = libvoikko.Voikko('fi')

    wordclasses = [
        ('noun', is_noun, get_baseform),
        ('verb', is_verb, get_wordbase),
        ('adj', is_adj, get_baseform),
        #('pron', is_pron, get_baseform),
        ('adv', is_adv, get_baseform),
    ]

    lemmarules = {
        "punct": [
            ["“", "\""],
            ["”", "\""],
            ["‘", "'"],
            ["’", "'"]
        ]
    }
    lemmaexceptions = {}

    lines = (line.strip().split(' ') for line in sys.stdin)
    word_counts = [(x[1], int(x[0])) for x in lines]
    word_counts_dict = dict(word_counts)

    for (spacyclass, voikko_filter, get_lemma) in wordclasses:
        words_with_lemmas = list(
            voikko_lemmas(word_counts, voikko_filter, get_lemma, word_counts_dict, v))
        motifs = find_motifs(words_with_lemmas)

        most_common_motifs = [
            {'rule': x[0], 'freq': x[1]}
            for x in takewhile(lambda x: x[1] >= min_freq, motifs.most_common())
        ]

        precision_motifs = estimate_precision(most_common_motifs, words_with_lemmas)
        precision_motifs = sorted(precision_motifs, key=lambda x: (x['precision'], x['freq']), reverse=True)

        with (output / f'{spacyclass}_motifs.json').open('w', encoding='utf-8') as fp:
            json.dump(precision_motifs, fp=fp, indent=2, ensure_ascii=False)

        rules = [
            m['rule'] for m in precision_motifs
            if m['precision'] > 0.80
        ]
        rules = sorted(rules, key=lambda r: len(r[0]), reverse=True)
        lemmarules[spacyclass] = rules

        exc = find_exceptions(words_with_lemmas, rules)
        exc = sorted(exc, key=lambda x: x[0])
        lemmaexceptions[spacyclass] = OrderedDict(exc)

    with (output / 'fi_lemma_rules.json').open('w', encoding='utf-8') as fp:
        json.dump(lemmarules, fp=fp, indent=2, ensure_ascii=False)

    with (output / 'fi_lemma_exc.json').open('w', encoding='utf-8') as fp:
        json.dump(lemmaexceptions, fp=fp, indent=2, ensure_ascii=False)


def voikko_lemmas(words, word_class_filter, baseform, freqs, v):
    for (word, f) in words:
        lemmas = [
            baseform(x) for x in v.analyze(word)
            if (word_class_filter(x) and not is_pure_num(x))
        ]
        lemmas = sorted(lemmas, key=lambda w: freqs.get(w, 0), reverse=True)

        if lemmas:
            yield (word, lemmas[0], f)


def is_pure_num(analysis):
    structure = analysis.get('STRUCTURE')
    return structure and pure_num_re.match(structure) is not None


def find_motifs(words_with_lemmas):
    motifs = Counter()
    for word, lemma, freq in words_with_lemmas:
        if word.lower() != lemma.lower():
            cpl = common_prefix_length(word, lemma)
            if cpl > 0:
                motifs[(word[cpl:], lemma[cpl:])] += freq

    return motifs


def common_prefix_length(x, y):
    for i, (a, b) in enumerate(zip(x, y)):
        if a != b:
            return i

    return min(len(x), len(y))


def estimate_precision(motifs, words):
    match = {}
    success = {}
    for word, lemma, freq in words:
        for motif in motifs:
            rule = tuple(motif['rule'])
            old, new = rule
            if word.endswith(old):
                match[rule] = match.get(rule, 0) + 1
                form = word[:-len(old)] + new
                if form.lower() == lemma.lower():
                    success[rule] = success.get(rule, 0) + 1

    res = []
    for motif in motifs:
        rule = tuple(motif['rule'])
        if rule in match:
            precision = success.get(rule, 0)/match.get(rule, 0)
        else:
            precision = 0

        res.append({
            'rule': motif['rule'],
            'freq': motif['freq'],
            'precision': precision
        })

    return res


def find_exceptions(words_with_lemmas, rules):
    lemmatizer = Lemmatizer(Lookups())

    for (word, expected_lemma, _) in words_with_lemmas:
        observed_lemma = lemmatizer.lemmatize(word, {}, {}, rules)[0]
        if observed_lemma.lower() != expected_lemma.lower():
            yield (word.lower(), [expected_lemma.lower()])


def is_noun(analysis):
    return analysis.get('CLASS') == 'nimisana'


def is_verb(analysis):
    plain = analysis.get('CLASS') in ['teonsana', 'kieltosana']
    # liittotempukset: "on haistanut", "oli nähnyt"
    tempus = (
        analysis.get('CLASS') == 'laatusana' and
        analysis.get('PARTICIPLE') in ['past_active', 'past_passive']
    )
    start_with_hyphen = analysis.get('BASEFORM', '').startswith('-')
    return (plain or tempus) and not start_with_hyphen


def is_adj(analysis):
    return (
        analysis.get('CLASS') in ['laatusana', 'nimisana_laatusana'] and
        analysis.get('PARTICIPLE') != 'past_active'
    )


def is_pron(analysis):
    return analysis.get('CLASS') == 'asemosana'


def is_adv(analysis):
    return analysis.get('CLASS') == 'seikkasana'


def get_baseform(analysis):
    return analysis.get('BASEFORM')


def get_wordbase(analysis):
    wordbases = analysis.get('WORDBASES', '')
    m = parenthesis_re.search(wordbases)
    if m:
        return m.group(1).replace('=', '')
    else:
        return analysis.get('BASEFORM')


if __name__ == '__main__':
    plac.call(main)
