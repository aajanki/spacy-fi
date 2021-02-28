import re
import sys


def main():
    for line in sys.stdin:
        if line == '\n' or line.startswith('#'):
            sys.stdout.write(line)
        else:
            columns = line.rstrip('\n').split('\t')
            token_id = columns[0]
            orth = columns[1]
            lemma = columns[2]
            upos = columns[3]
            xpos = columns[4]

            if '-' in token_id:
                # Skip multiword tokens.
                #
                # Assert that the UD input has undefined data on
                # multiword tokens.
                assert all(x == '_' for x in columns[2:])
                continue

            columns[2] = fix_compund_word_lemmas(orth, lemma)

            # error in the data?
            if xpos == 'Adj':
                columns[4] = 'A'

            # The fine-grained tags are some times more coarse than
            # the "coarse-grained" tags
            if upos == 'SCONJ':
                columns[4] = 'SC'
            elif upos == 'PROPN':
                columns[4] = 'Propn'
            elif upos == 'AUX':
                columns[4] = 'Aux'

            columns[9] = 'O'

            sys.stdout.write('\t'.join(columns))
            sys.stdout.write('\n')


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
    main()
