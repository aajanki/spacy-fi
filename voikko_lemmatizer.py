from spacy.lemmatizer import Lemmatizer
from spacy.symbols import NOUN, VERB, ADJ, ADV, PROPN
from voikko import libvoikko


class VoikkoLemmatizer(Lemmatizer):
    def __init__(self, *args, **kwargs):
        super(VoikkoLemmatizer, self).__init__(*args, **kwargs)
        self.voikko = libvoikko.Voikko('fi')

    def __call__(self, string, univ_pos, morphology=None):
        if univ_pos in (NOUN, 'NOUN', 'noun'):
            voikko_pos = 'nimisana'
        elif univ_pos in (VERB, 'VERB', 'verb'):
            voikko_pos = 'teonsana'
        elif univ_pos in (ADJ, 'ADJ', 'adj'):
            voikko_pos = 'laatusana'
        elif univ_pos in (ADV, 'ADV', 'adv'):
            voikko_pos = 'seikkasana'
        elif univ_pos in (PROPN, 'PROPN'):
            return [string]
        else:
            return [string.lower()]

        analyses = self.voikko.analyze(string)
        filtered_by_pos = [
            x.get('BASEFORM') for x in analyses
            if x.get('CLASS') == voikko_pos and x.get('BASEFORM')
        ]
        if filtered_by_pos:
            return filtered_by_pos
        elif analyses:
            return [x.get('BASEFORM') for x in analyses if x.get('BASEFORM')]
        else:
            return [string.lower()]
