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

    words = (line.strip().split(' ', 1)[1] for line in sys.stdin)
    words = (x.strip(strip_punct) for x in words)
    lemmas = voikko_lemmas(words, v)
    lemmas = OrderedDict(sorted(lemmas))
    json.dump(lemmas, fp=sys.stdout, indent=2, ensure_ascii=False)


def voikko_lemmas(words, v):
    for word in words:
        analyses = (
            x for x in v.analyze(word)
            if x.get('BASEFORM') and not is_pure_num(x)
        )

        try:
            lemma = next(analyses).get('BASEFORM')
        except StopIteration:
            continue

        if lemma and word.lower() != lemma.lower():
            yield (word, lemma)


def is_pure_num(analysis):
    structure = analysis.get('STRUCTURE')
    if not structure:
        return False

    return pure_num_re.match(structure) is not None


if __name__ == '__main__':
    main()
