import sys


def main():
    keep_tags = ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'O']

    sentence_boundary = True
    for line in sys.stdin:
        if line == '\n' or line == '\t\t\n':
            if not sentence_boundary:
                sys.stdout.write('\n')
            sentence_boundary = True
        elif line in ['<HEADLINE>\t\t\n', '<BODY>\t\t\n', '<INGRESS>\t\t\n']:
            sentence_boundary = True
        else:
            sentence_boundary = False

            cols = line.split('\t')
            tag = cols[1] if cols[1] in keep_tags else 'O'
            outcols = [cols[0], tag]

            sys.stdout.write('\t'.join(outcols))
            sys.stdout.write('\n')


if __name__ == '__main__':
    main()
