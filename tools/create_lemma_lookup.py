import gzip
import json
import re
import sys
from tqdm import tqdm
from voikko import libvoikko

pure_num_re = re.compile(r'^-?=q+(:p+)$')


def main():
    infile, outfile = sys.argv[1:3]
    voikko = libvoikko.Voikko('fi')

    with gzip.open(infile, 'rt', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    lemmas = {}
    with gzip.open(infile, 'rt') as f:
        for line in tqdm(f, total=line_count):
            word = line.strip().split(' ', 1)[1]
            analyses = voikko.analyze(word)

            if analyses:
                lemma = analyses[0].get('BASEFORM')
                if (lemma and
                    lemma.lower() != word.lower() and
                    not is_pure_num(analyses[0])
                ):
                    lemmas[word] = lemma

    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(lemmas, f, indent=2, ensure_ascii=False)


def is_pure_num(analysis):
    structure = analysis.get('STRUCTURE')
    if not structure:
        return False
    return pure_num_re.match(structure) is not None


if __name__ == '__main__':
    main()
