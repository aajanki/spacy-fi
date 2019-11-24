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


@plac.annotations(
    min_freq=('Minimum rule frequency', 'option', 'f', int),
    output=('Output path', 'option', None, Path)
)
def main(min_freq=10000, output=Path('data/lemma')):
    v = libvoikko.Voikko('fi')

    wordclasses = [
        (['nimisana'], 'noun'),
        (['teonsana', 'kieltosana'], 'verb'),
        (['laatusana', 'nimisana_laatusana'], 'adj'),
        #(['asemosana'], 'pron'),
        #('seikkasana', 'adv'),
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

    for (voikkoclasses, spacyclass) in wordclasses:
        words_with_lemmas = list(voikko_lemmas(word_counts, voikkoclasses, word_counts_dict, v))
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
            if m['precision'] > 0.1
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


def voikko_lemmas(words, wordclasses, freqs, v):
    for (word, f) in words:
        analyses = [
            x.get('BASEFORM') for x in v.analyze(word)
            if (x.get('CLASS') in wordclasses and
                x.get('BASEFORM') and
                not is_pure_num(x))
        ]
        analyses = sorted(analyses, key=lambda w: freqs.get(w, 0), reverse=True)

        if analyses:
            lemma = analyses[0]
            yield (word, lemma, f)


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
    lemmatizer = Lemmatizer(Lookups())

    for (word, expected_lemma, _) in words_with_lemmas:
        observed_lemma = lemmatizer.lemmatize(word, {}, {}, rules)[0]
        if observed_lemma.lower() != expected_lemma.lower():
            yield (word.lower(), [expected_lemma.lower()])


if __name__ == '__main__':
    plac.call(main)
