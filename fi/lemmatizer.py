# coding: utf8
from __future__ import unicode_literals

import json
import re
from collections import OrderedDict
from itertools import chain

from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from spacy.symbols import NOUN, VERB, ADJ, PUNCT, PROPN, ADV, NUM, PRON, AUX
from voikko import libvoikko


class FinnishLemmatizer(Lemmatizer):
    compound_re = re.compile(r"\+(\w+)(?:\(\+?[\w=]+\))?")
    minen_re = re.compile(r"\b(\w+)\[Tn4\]mi")
    sti_re = re.compile(r"\b(\w+)\[Ssti\]sti")
    ny_re = re.compile(r"\[X\]\[\w+\]\[Ny\](\w+)")
    voikko_pos_to_upos = {
        "nimisana": "noun",
        "teonsana": "verb",
        "laatusana": "adj",
        "nimisana_laatusana": "adj",
        "seikkasana": "adv",
        "lukusana": "num",
        "nimi": "propn",
        "etunimi": "propn",
        "sukunimi": "propn",
        "paikannimi": "propn",
        "asemosana": "pron",
    }

    # Use singular pronoun as lemmas (similar to in universal
    # dependencies)
    pron_baseform_exceptions = {
        'me': 'minä',
        'te': 'sinä',
        'he': 'hän',
        'nämä': 'tämä',
        'nuo': 'tuo',
        'ne': 'se',
        'ken': 'kuka',
    }

    def __init__(self, lookups, *args, **kwargs):
        super(FinnishLemmatizer, self).__init__(lookups, *args, **kwargs)
        self.voikko = libvoikko.Voikko("fi")

    def __call__(self, string, univ_pos, morphology=None):
        """Lemmatize a string.

        string (unicode): The string to lemmatize, e.g. the token text.
        univ_pos (unicode / int): The token's universal part-of-speech tag.
        morphology (dict): The token's morphological features following the
            Universal Dependencies scheme.
        RETURNS (list): The available lemmas for the string.
        """
        if univ_pos in (NOUN, "NOUN", "noun"):
            univ_pos = "noun"
        elif univ_pos in (VERB, "VERB", "verb", AUX, "AUX", "aux"):
            univ_pos = "verb"
        elif univ_pos in (ADJ, "ADJ", "adj"):
            univ_pos = "adj"
        elif univ_pos in (ADV, "ADV", "adv"):
            univ_pos = "adv"
        elif univ_pos in (NUM, "NUM", "num"):
            univ_pos = "num"
        elif univ_pos in (PROPN, "PROPN", "propn"):
            univ_pos = "propn"
        elif univ_pos in (PRON, "PRON", "pron"):
            univ_pos = "pron"
        elif univ_pos in (PUNCT, "PUNCT", "punct"):
            return [string]
        else:
            return [string.lower()]

        index_table = self.lookups.get_table("lemma_index", {})
        exc_table = self.lookups.get_table("lemma_exc", {})
        rules_table = self.lookups.get_table("lemma_rules", {})
        lemmas = self.lemmatize(
            string,
            index_table.get(univ_pos, {}),
            exc_table.get(univ_pos, {}),
            rules_table.get(univ_pos, {}),
            univ_pos,
        )
        return lemmas

    def lemmatize(self, string, index, exceptions, rules, univ_pos):
        # base of an inflected abbreviations: BBC:n, EU:ssa
        string = string.rsplit(":", 1)[0]
        
        # Lemmatize only the last part of hyphenated words: VGA-kaapelissa
        parts = string.rsplit("-", 1)
        
        lemma = self.lemmatize_compound(parts[-1], index, exceptions, rules, univ_pos)

        if len(parts) == 1:
            return lemma
        else:
            return [parts[0] + "-" + lemma[0]]

    def lemmatize_compound(self, string, index, exceptions, rules, univ_pos):
        orig = string
        oov_forms = []
        forms = []

        analyses = self.voikko.analyze(string)
        base_and_pos = list(chain.from_iterable([
            self._baseform_and_pos(x, string) for x in analyses
        ]))
        matching_pos = [x for x in base_and_pos if x[1] == univ_pos]
        if univ_pos == "adv" and analyses:
            oov_forms.append(self._normalize_adv(analyses[0], orig.lower()))
        elif matching_pos:
            forms.extend(x[0] for x in matching_pos)
        elif analyses:
            oov_forms.extend(x[0] for x in base_and_pos)

        forms = list(OrderedDict.fromkeys(forms))

        # Put exceptions at the front of the list, so they get priority.
        # This is a dodgy heuristic -- but it's the best we can do until we get
        # frequencies on this. We can at least prune out problematic exceptions,
        # if they shadow more frequent analyses.
        for exc in exceptions.get(orig.lower(), []):
            if exc not in forms:
                forms.insert(0, exc)
        if not forms:
            forms.extend(oov_forms)
        if not forms:
            forms.append(orig)
        return forms

    def _baseform_and_pos(self, analysis, orig):
        baseform = analysis.get("BASEFORM")
        voikko_class = analysis.get("CLASS")

        if (voikko_class == "teonsana" and
            analysis.get("MOOD") == "MINEN-infinitive"
        ):
            # MINEN infinitive
            form = self._fst_form(analysis, self.minen_re, "minen")
            if form:
                return [(form, "noun")]
            else:
                return [(baseform, "verb")]

        elif (voikko_class == "laatusana" and
              analysis.get("PARTICIPLE") in ["past_active",
                                             "past_passive",
                                             "present_active",
                                             "present_passive"]
        ):
            # VA, NUT and TU participles
            return [
                (self._wordbase(analysis), "verb"),
                (baseform, "adj")
            ]

        elif (voikko_class == "nimisana" and
              analysis.get("PARTICIPLE") == "agent"
        ):
            # agent participle
            return [(self._wordbase(analysis), "verb")]

        elif (voikko_class in ["laatusana", "lukusana"] and
              analysis.get("SIJAMUOTO") == "kerrontosti"
        ):
            form = self._fst_form(analysis, self.sti_re, "sti")
            if form:
                return [(form, "adv")]
            else:
                return [(baseform, self.voikko_pos_to_upos[voikko_class])]

        elif voikko_class == "seikkasana" and orig.endswith("itse"):
            return [(orig, "adv")]

        elif voikko_class == "asemosana":
            lemma = self.pron_baseform_exceptions.get(baseform, baseform)
            return [(lemma, self.voikko_pos_to_upos[voikko_class])]

        elif voikko_class in self.voikko_pos_to_upos:
            return [(baseform, self.voikko_pos_to_upos[voikko_class])]

        else:
            return [(baseform, None)]

    def _fst_form(self, analysis, stem_re, suffix):
        fstoutput = analysis.get("FSTOUTPUT")
        ny_match = self.ny_re.search(fstoutput)
        if ny_match:
            return ny_match.group(1)

        fst_match = stem_re.search(fstoutput)
        if not fst_match:
            return None

        stem = fst_match.group(1)
        compounds = self.compound_re.findall(analysis.get("WORDBASES"))
        if len(compounds) > 1:
            return "".join(compounds[:-1]) + stem + suffix
        else:
            return stem + suffix

    def _wordbase(self, analysis):
        wordbases = analysis.get("WORDBASES")
        num_bases = max(analysis.get("STRUCTURE", "").count("="), 1)

        i = 0
        forms = []
        for base in re.finditer(r"\+([^+]+)", wordbases):
            full_form = base.group(1)
            parentheses_match = re.search(r"(.+)\((.+)\)", full_form)
            if parentheses_match:
                k = parentheses_match.group(2).count("=") + 1
                if i < num_bases - k:
                    form = parentheses_match.group(1)
                else:
                    form = parentheses_match.group(2)
            else:
                form = full_form

            split = form.split("=")
            forms.extend(split)
            i += len(split)

            if i >= num_bases:
                break

        return ''.join(forms)

    def _normalize_adv(self, analysis, word):
        focus = analysis.get("FOCUS")
        kysymysliite = analysis.get("KYSYMYSLIITE")

        if focus and kysymysliite:
            k = 2
        elif focus or kysymysliite:
            k = 1
        else:
            k = 0
        for _ in range(k):
            if focus and word.endswith(focus):
                word = word[:-len(focus)]
            elif kysymysliite and (word.endswith("ko") or word.endswith("kö")):
                word = word[:-2]

        if analysis.get("POSSESSIVE") and not analysis.get("SIJAMUOTO"):
            return analysis.get("BASEFORM")
        else:
            return word
