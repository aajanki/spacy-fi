"""Convert a binary word2vec file to text format"""

import sys
from gensim.models import KeyedVectors


infile, outfile = sys.argv[1:3]

print('Reading binary word2vec file...')
wv = KeyedVectors.load_word2vec_format(infile, binary=True, unicode_errors='ignore')

print('Saving text word2vec file...')
wv.save_word2vec_format(outfile, binary=False)
