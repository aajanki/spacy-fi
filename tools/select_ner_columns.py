import sys


def main():
    infile, outfile = sys.argv[1:3]

    sentence_boundary = True
    with open(infile) as inf, open(outfile, 'w') as outf:
        for line in inf:
            if line == '\n' or line == '\t\t\n':
                if not sentence_boundary:
                    outf.write('\n')
                sentence_boundary = True
            elif line in ['<HEADLINE>\t\t\n', '<BODY>\t\t\n', '<INGRESS>\t\t\n']:
                sentence_boundary = True
            else:
                sentence_boundary = False

                cols = line.split('\t')
                outf.write('\t'.join(cols[:2]))
                outf.write('\n')


if __name__ == '__main__':
    main()
