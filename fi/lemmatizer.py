import re
from collections import OrderedDict
from itertools import chain
from typing import Callable, Iterable, List, Optional
from spacy.errors import Errors
from spacy.lang.fi import Finnish
from spacy.language import Language
from spacy.lookups import Lookups, load_lookups
from spacy.pipeline.lemmatizer import Lemmatizer
from spacy.symbols import NOUN, VERB, ADJ, PROPN, ADV, NUM, PRON, AUX
from spacy.tokens import Token
from spacy.training import Example
from spacy.vocab import Vocab
from thinc.api import Model
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

    def __init__(self, vocab: Vocab, name: str = "lemmatizer", overwrite: bool = False) -> None:
        super().__init__(vocab, model=None, name=name, mode="voikko", overwrite=overwrite)
        self.voikko = libvoikko.Voikko("fi")

    def initialize(
        self,
        get_examples: Optional[Callable[[], Iterable[Example]]] = None,
        *,
        nlp: Optional[Language] = None,
        lookups: Optional[Lookups] = None,
    ):
        """Initialize the lemmatizer and load in data.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        lookups (Lookups): The lookups object containing the (optional) tables
            such as "lemma_rules", "lemma_index", "lemma_exc" and
            "lemma_lookup". Defaults to None.
        """
        required_tables = ["lemma_exc"]
        if lookups is None:
            #logger.debug("Lemmatizer: loading tables from spacy-lookups-data")
            lookups = load_lookups(lang=self.vocab.lang, tables=required_tables)
        self.lookups = lookups
        self._validate_tables(Errors.E1004)

    def voikko_lemmatize(self, token: Token) -> List[str]:
        """Lemmatize one token using voikko.

        token (Token): The token to lemmatize.
        RETURNS (list): The available lemmas for the string.
        """
        if token.pos == NOUN:
            univ_pos = "noun"
        elif token.pos in (VERB, AUX):
            univ_pos = "verb"
        elif token.pos == ADJ:
            univ_pos = "adj"
        elif token.pos == ADV:
            univ_pos = "adv"
        elif token.pos == NUM:
            univ_pos = "num"
        elif token.pos == PROPN:
            univ_pos = "propn"
        elif token.pos == PRON:
            univ_pos = "pron"
        else:
            return [token.orth_.lower()]

        exc_table = self.lookups.get_table("lemma_exc", {})
        pos_exc_table = exc_table.get(univ_pos, {})
        return self._lemmatize_one_word(token.orth_, pos_exc_table, univ_pos)

    def _lemmatize_one_word(self, string, exceptions, univ_pos):
        # Lemma of inflected abbreviations: BBC:n, EU:ssa
        string = string.rsplit(":", 1)[0]
        
        # Lemmatize only the last part of hyphenated words: VGA-kaapelissa
        parts = string.rsplit("-", 1)
        
        lemma = self._lemmatize_compound(parts[-1], exceptions, univ_pos)

        if len(parts) == 1:
            return lemma
        else:
            return [parts[0] + "-" + lemma[0]]

    def _lemmatize_compound(self, string, exceptions, univ_pos):
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


@Finnish.factory(
    "lemmatizer",
    assigns=["token.lemma"],
    default_config={"model": None, "mode": "rule", "overwrite": False},
    default_score_weights={"lemma_acc": 1.0},
)
def make_lemmatizer(
    nlp: Language, model: Optional[Model], name: str, mode: str, overwrite: bool = False
):
    return FinnishLemmatizer(nlp.vocab, model, name, overwrite=overwrite)
