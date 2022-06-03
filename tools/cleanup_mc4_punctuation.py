import re
import sys
from spacy.lang.char_classes import ALPHA, CONCAT_QUOTES

for line in sys.stdin:
    line = re.sub(r'(?<=[-+*/!?%(),:;<>€$£])\.(?=\w)', '. ', line)
    line = re.sub(r'\.(?=[-!?%(),:;<>€$£])', '. ', line)
    line = re.sub(r'(?<=[{a}]{{3}})\.(?=[{a}]{{3}})'.format(a=ALPHA), '. ', line)
    line = re.sub(r'(?<=[{a}])\.(?=\d)'.format(a=ALPHA), '. ', line)
    line = re.sub(r'(?<=\w)\.(?=[{q}])'.format(q=CONCAT_QUOTES), '. ', line)
    line = re.sub(r'(?<=[{q}])\.(?=\w)'.format(q=CONCAT_QUOTES), '. ', line)
    line = re.sub(r'(?<=\d)\.(?=[{a}])'.format(a=ALPHA), '. ', line)
    line = re.sub(r'(?<=[{a}]{{2}})\.(?=\d)'.format(a=ALPHA), '. ', line)
    line = re.sub(r' \.(?=[{a}])'.format(a=ALPHA), '. ', line)

    sys.stdout.write(line)
