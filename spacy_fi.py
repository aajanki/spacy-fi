"""A wrapper around spacy cli tools for injecting the extended Finnish data"""

import sys
import plac
from spacy import cli, util
from fi.fi import FinnishEx


def lexeme_lookups():
    return {
        'lexeme_prob': 'data/lexeme/fi_lexeme_prob.json',
        'lexeme_settings': 'data/lexeme/fi_lexeme_settings.json',
    }


if __name__ == '__main__':
    util.registry.lookups.register('fi_extra', func=lexeme_lookups())
    util.set_lang_class('fi', FinnishEx)

    command_name = sys.argv[1].replace('-', '_')
    command = getattr(cli, command_name)
    plac.call(command, sys.argv[2:])
