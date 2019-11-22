import json
import re
import sys
import plac
from collections import OrderedDict
from itertools import chain, islice
from pathlib import Path
from voikko import libvoikko

pure_num_re = re.compile(r'^-?=q+(:p+)$')


@plac.annotations(
    rulefile=('The path to fi_lemma_rules.json', 'positional', None, Path)
)
def main(rulefile):
    v = libvoikko.Voikko('fi')

    rules = load_rules(rulefile)

    lines = (line.strip().split(' ') for line in sys.stdin)
    word_counts = ((x[1], int(x[0])) for x in lines)
    frequent_words = (w[0] for w in word_counts if w[1] >= 10)
    words_with_lemmas = voikko_lemmas(frequent_words, v)

    lookup = OrderedDict(sorted(lemma_exceptions(words_with_lemmas, rules)))
    json.dump(lookup, fp=sys.stdout, indent=2, ensure_ascii=False)


def lemma_exceptions(words_with_lemmas, rules):
    for word, lemmas in words_with_lemmas:
        found_match = False
        for rule in rules:
            if word.endswith(rule[0]):
                form = word[:-len(rule[0])] + rule[1]
                if form.lower() in lemmas:
                    found_match = True
                    break

        if not found_match and word.lower() != lemmas[0]:
            yield (word, lemmas[0])


def load_rules(rulefile):
    rules = json.load(rulefile.open())
    return list(chain.from_iterable(rules.values()))


def voikko_lemmas(words, v):
    for word in words:
        analyses = (x for x in v.analyze(word) if x.get('BASEFORM'))
        analyses = (x for x in analyses if not is_pure_num(x))
        lemmas = [x.get('BASEFORM').lower() for x in analyses]
        if lemmas:
            yield (word, lemmas)


def is_pure_num(analysis):
    structure = analysis.get('STRUCTURE')
    if not structure:
        return False

    return pure_num_re.match(structure) is not None


if __name__ == '__main__':
    plac.call(main)
