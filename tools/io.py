import bz2
import gzip
import sys
from pathlib import Path


def open_input(p: Path):
    if str(p) == '-':
        return sys.stdin
    elif p.suffix == '.bz2':
        return bz2.open(p, 'rt', encoding='utf-8')
    elif p.suffix == '.gz':
        return gzip.open(p, 'rt', encoding='utf-8')
    else:
        return open(p, 'r')


def open_output(p: Path):
    if str(p) == '-':
        return sys.stdout
    elif p.suffix == '.bz2':
        return bz2.open(p, 'wt', encoding='utf-8')
    elif p.suffix == '.gz':
        return gzip.open(p, 'wt', encoding='utf-8')
    else:
        return open(p, 'w')
