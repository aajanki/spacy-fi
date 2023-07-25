import re
import sys
import typer
from pathlib import Path

TOKEN_ID = 0
ORTH = 1
LEMMA = 2
UPOS = 3
XPOS = 4
DEPREL = 7
DEPS = 8
MISC = 9

def main(
        input_file: Path = typer.Argument(..., help='Input file'),
        output_file: Path = typer.Argument(..., help='Output file'),
        trainset: bool = typer.Option(False, help='Extra preprocessing for the training set'),
):
    with open(input_file) as inf, open(output_file, 'w') as outf:
        for line in inf:
            if line == '\n' or line.startswith('#'):
                outf.write(line)
            else:
                columns = line.rstrip('\n').split('\t')

                if '-' in columns[TOKEN_ID]:
                    # Skip multiword tokens.
                    #
                    # Assert that the UD input has undefined data on
                    # multiword tokens.
                    assert all(x == '_' for x in columns[2:])
                    continue

                columns[LEMMA] = fix_compund_word_lemmas(columns[ORTH], columns[LEMMA])

                # error in the data?
                if columns[XPOS] == 'Adj':
                    columns[XPOS] = 'A'

                # The fine-grained tags are some times more coarse than
                # the "coarse-grained" tags
                if columns[UPOS] == 'SCONJ':
                    columns[XPOS] = 'SC'
                elif columns[UPOS] == 'PROPN':
                    columns[XPOS] = 'Propn'
                elif columns[UPOS] == 'AUX':
                    columns[XPOS] = 'Aux'

                columns[MISC] = 'O'

                if trainset:
                    # There are too few 'nsubj:outer's, 'dislocated's
                    # and 'goeswith's for learning. Replace them in
                    # the training set to avoid wasting model capacity
                    # on them. 'nsubj:outer' is replaced with
                    # 'nsubj:cop' like they used to be in
                    # UD-Finnish-TDT before October 2022. Others are
                    # replaced by general 'dep's.
                    if columns[DEPREL] == 'nsubj:outer':
                        columns[DEPREL] = 'nsubj:cop'
                        columns[DEPS] = columns[DEPS].replace('nsubj:outer', 'nsubj:cop')
                    elif columns[DEPREL] == 'dislocated':
                        columns[DEPREL] = 'dep'
                        columns[DEPS] = columns[DEPS].replace('dislocated', 'dep')
                    elif columns[DEPREL] == 'goeswith':
                        columns[DEPREL] = 'dep'
                        columns[DEPS] = columns[DEPS].replace('goeswith', 'dep')

                outf.write('\t'.join(columns))
                outf.write('\n')


def fix_compund_word_lemmas(orth, lemma):
    new = re.sub(r'(\w+?)#(\w+?)', replace_compound_separator, lemma)
    new = re.sub(r'#(?=[-.!?])', '', new)
    new = re.sub(r'(?<=[-.!?])#', '', new)
    if re.match(r'#\w', new):
        new = '-' + new[1:]
    return new


def replace_compound_separator(m):
    a = m.group(1)[-1]
    b = m.group(2)[0]

    vowels = 'aeiouyäö'
    duplicate_vowel = (a.lower() == b.lower()) and (a.lower() in vowels)

    if a.isdigit() and len(m.group(2)) <= 2:
        return m.group(1) + m.group(2)
    elif a.isdigit() or b.isdigit() or a.isupper() or b.isupper() or duplicate_vowel:
        return m.group(1) + '-' + m.group(2)
    else:
        return m.group(1) + m.group(2)


if __name__ == '__main__':
    typer.run(main)
