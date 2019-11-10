"""A wrapper around spacy cli tools for injecting the extended Finnish data"""

import sys
import plac
from spacy import cli, util
from fi import FinnishEx


if __name__ == '__main__':
    util.set_lang_class('fi', FinnishEx)

    command_name = sys.argv[1].replace('-', '_')
    command = getattr(cli, command_name)
    plac.call(command, sys.argv[2:])
