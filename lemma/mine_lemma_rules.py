import json
import sys
import plac
from collections import Counter, OrderedDict
from itertools import takewhile
from pathlib import Path
from voikko import libvoikko


@plac.annotations(
    min_freq=('Minimum rule frequency', 'option', 'f', int),
    output=('Output path', 'option', None, Path)
)
def main(min_freq=10000, output=Path('data/lemma')):
    v = libvoikko.Voikko('fi')

    wordclasses = [
        ('nimisana', 'noun'),
        ('teonsana', 'verb'),
        ('laatusana', 'adj'),
        ('asemosana', 'pron'),
    ]

    lemmarules = {
        "punct": [
            ["“", "\""],
            ["”", "\""],
            ["‘", "'"],
            ["’", "'"]
        ]
    }
    exceptions = {}

    lines = (line.strip().split(' ') for line in sys.stdin)
    word_counts = ((x[1], int(x[0])) for x in lines)

    for (voikkoclass, spacyclass) in wordclasses:
        words_with_lemmas = list(voikko_lemmas(word_counts, voikkoclass, v))
        motifs = find_motifs(words_with_lemmas)

        most_common_motifs = [
            {'rule': x[0], 'freq': x[1]}
            for x in takewhile(lambda x: x[1] >= min_freq, motifs.most_common())
        ]

        precision_motifs = estimate_precision(most_common_motifs, words_with_lemmas)
        precision_motifs = sorted(precision_motifs, key=lambda x: (x['precision'], x['freq']), reverse=True)

        with (output / f'{spacyclass}_motifs.json').open('w', encoding='utf-8') as fp:
            json.dump(precision_motifs, fp=fp, indent=2, ensure_ascii=False)

        lemmarules[spacyclass] = [
            m['rule'] for m in precision_motifs
            if m['precision'] > 0.2
        ]

        exceptions[spacyclass] = \
            OrderedDict(find_exceptions(words_with_lemmas, lemmarules[spacyclass]))

    with (output / 'fi_lemma_rules.json').open('w', encoding='utf-8') as fp:
        json.dump(lemmarules, fp=fp, indent=2, ensure_ascii=False)

    with (output / 'fi_lemma_exc.json').open('w', encoding='utf-8') as fp:
        json.dump(exceptions, fp=fp, indent=2, ensure_ascii=False)


def voikko_lemmas(word_counts, wordclass, v):
    for (word, freq) in word_counts:
        analyses = [x for x in v.analyze(word)
                    if x.get('CLASS') == wordclass and x.get('BASEFORM')]
        if analyses:
            lemma = analyses[0].get('BASEFORM').lower()
            yield (word.lower(), lemma, freq)


def find_motifs(words_with_lemmas):
    motifs = Counter()
    for word, lemma, freq in words_with_lemmas:
        if word != lemma:
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
                if form == lemma:
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
    for (word, lemma, _) in words_with_lemmas:
        match = any(rule_matches(r, word, lemma) for r in rules)
        if not match:
            yield (word.lower(), [lemma.lower()])


def rule_matches(rule, word, lemma):
    word = word.lower()
    if word.endswith(rule[0]):
        form = word[:-len(rule[0])] + rule[1]
        if form.lower() == lemma.lower():
            return True

    return False


if __name__ == '__main__':
    plac.call(main)
