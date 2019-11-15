import sys

for line in sys.stdin:
    if line == '\n' or line.startswith('#'):
        sys.stdout.write(line)
    else:
        columns = line.rstrip('\n').split('\t')
        upos = columns[3]
        xpos = columns[4]

        # error in the data?
        if xpos == 'Adj':
            columns[4] = 'A'

        # The fine-grained tags are some times more coarse than
        # the "coarse-grained" tags
        if upos == 'SCONJ':
            columns[4] = 'SC'
        elif upos == 'PROPN':
            columns[4] = 'Propn'

        columns[9] = 'O'

        sys.stdout.write('\t'.join(columns))
        sys.stdout.write('\n')
