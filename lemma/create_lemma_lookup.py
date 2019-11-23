import json
import re
import string
import sys
from collections import OrderedDict
from voikko import libvoikko

pure_num_re = re.compile(r'^-?=q+(:p+)$')


def main():
    v = libvoikko.Voikko('fi')
    strip_punct = ''.join(x for x in string.punctuation if x != '-')

    lines = (line.strip().split(' ', 1) for line in sys.stdin)
    word_counts = [(x[1], int(x[0])) for x in lines]
    words = (x[0].strip(strip_punct) for x in word_counts)
    lemmas = voikko_lemmas(words, dict(word_counts), v)
    lemmas = OrderedDict(sorted(lemmas))
    json.dump(lemmas, fp=sys.stdout, indent=2, ensure_ascii=False)


def voikko_lemmas(words, freqs, v):
    for word in words:
        analyses = [
            x.get('BASEFORM') for x in v.analyze(word)
            if x.get('BASEFORM') and not is_pure_num(x)
        ]
        analyses = sorted(analyses, key=lambda w: freqs.get(w, 0), reverse=True)

        if analyses:
            lemma = analyses[0]

            if lemma and word.lower() != lemma.lower():
                yield (word, lemma)


def is_pure_num(analysis):
    structure = analysis.get('STRUCTURE')
    if not structure:
        return False

    return pure_num_re.match(structure) is not None


if __name__ == '__main__':
    main()
