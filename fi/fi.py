import re
import srsly
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union
from spacy import util
from spacy.errors import Errors
from spacy.lang.fi import Finnish, FinnishDefaults
from spacy.language import Language
from spacy.lookups import Lookups, load_lookups
from spacy.pipeline.pipe import Pipe
from spacy.scorer import Scorer
from spacy.symbols import ADJ, ADP, ADV, AUX, CCONJ, INTJ, NOUN, NUM, PROPN
from spacy.symbols import PRON, PUNCT, SCONJ, SPACE, SYM, VERB, X
from spacy.symbols import acl, aux, cc, conj, cop, obj
from spacy.tokens import Doc, Span, Token
from spacy.training import Example, validate_examples
from spacy.util import SimpleFrozenList
from spacy.vocab import Vocab
from voikko import libvoikko
from zipfile import ZipFile


class MorphologizerLemmatizer(Pipe):
    """Pipeline component that assigns morphological features and lemmas to Docs.

    POS tags must have been assigned prior to this pipeline component.

    The actual morphological analysis is done by libvoikko.
    """
    compound_re = re.compile(r"\+(\w+)(?:\(\+?[\w=]+\))?")
    minen_re = re.compile(r"\b(\w+)\[Tn4\]mi")
    ny_re = re.compile(r"\[X\]\[\w+\]\[Ny\](\w+)")
    roman_numeral_structure_re = re.compile(r"=j+|=q+")
    voikko_cases = {
        "nimento":     "Case=Nom",
        "omanto":      "Case=Gen",
        "kohdanto":    "Case=Acc",
        "olento":      "Case=Ess",
        "osanto":      "Case=Par",
        "tulento":     "Case=Tra",
        "sisaolento":  "Case=Ine",
        "sisaeronto":  "Case=Ela",
        "sisatulento": "Case=Ill",
        "ulkoolento":  "Case=Ade",
        "ulkoeronto":  "Case=Abl",
        "ulkotulento": "Case=All",
        "vajanto":     "Case=Abe",
        "seuranto":    "Case=Com",
        "keinonto":    "Case=Ins",
        "kerrontosti": "Case=Nom"  # Should never occur. "kerrontosti"
                                   # should only appear on ADVs, which
                                   # don't have cases.
    }
    voikko_classes_by_pos = {
        ADJ:   frozenset(["laatusana", "nimisana_laatusana"]),
        ADP:   frozenset(["nimisana", "seikkasana", "suhdesana"]),
        ADV:   frozenset(["seikkasana"]),
        AUX:   frozenset(["teonsana", "kieltosana"]),
        CCONJ: frozenset(["sidesana"]),
        INTJ:  frozenset(["huudahdussana"]),
        NOUN:  frozenset(["nimisana", "nimisana_laatusana", "lyhenne"]),
        NUM:   frozenset(["lukusana"]),
        PRON:  frozenset(["asemosana", "nimisana", "nimisana_laatusana"]),
        PROPN: frozenset(["nimi", "etunimi", "sukunimi", "paikannimi"]),
        SCONJ: frozenset(["sidesana"]),
        VERB:  frozenset([]), # Would be "teonsana" but
                              # MINEN-infinitives are treated as noun.
                              # See _analysis_has_compatible_pos()
        SYM:   frozenset([]),
        X:     frozenset([])
    }
    affix_to_sijamuoto = {
        "n":    "omanto",
        "na":   "olento",
        "nä":   "olento",
        "a":    "osanto",
        "ä":    "osanto",
        "ta":   "osanto",
        "tä":   "osanto",
        "ksi":  "tulento",
        "ssa":  "sisaolento",
        "ssä":  "sisaolento",
        "sta":  "sisaeronto",
        "stä":  "sisaeronto",
        "han":  "sisatulento",
        "hin":  "sisatulento",
        "hun":  "sisatulento",
        "seen": "sisatulento",
        "siin": "sisatulento",
        "lla":  "ulkoolento",
        "llä":  "ulkoolento",
        "lta":  "ulkoeronto",
        "ltä":  "ulkoeronto",
        "lle":  "ulkotulento",
        "tta":  "vajanto",
        "ttä":  "vajanto",
    }
    possessive_suffixes = {
        "1s": ["ni"],
        "2s": ["si"],
        "1p": ["mme"],
        "2p": ["nne"],
        "3": ["nsa", "nsä", "an", "en", "in" "on", "un", "yn", "än", "ön"],
    }
    voikko_degree = {
        "positive":    "Degree=Pos",
        "comparative": "Degree=Cmp",
        "superlative": "Degree=Sup"
    }
    voikko_mood = {
        "A-infinitive":  "InfForm=1",
        "E-infinitive":  "InfForm=2",
        "MA-infinitive": "InfForm=3",
        "indicative":    "Mood=Ind",
        "conditional":   "Mood=Cnd",
        "potential":     "Mood=Pot",
        "imperative":    "Mood=Imp"
    }
    voikko_part_form = {
        "past_active":     "PartForm=Past",
        "past_passive":    "PartForm=Past",
        "present_active":  "PartForm=Pres",
        "present_passive": "PartForm=Pres",
        "agent":           "PartForm=Agt"
    }
    voikko_tense = {
        "present_active":    "Tense=Pres",
        "present_passive":   "Tense=Pres",
        "present_simple":    "Tense=Pres",
        "past_active":       "Tense=Past",
        "past_passive":      "Tense=Past",
        "past_imperfective": "Tense=Past"
    }
    pron_types = {
        "minä":       "Prs",
        "sinä":       "Prs",
        "hän":        "Prs",
        "me":         "Prs",
        "te":         "Prs",
        "he":         "Prs",

        "tämä":       "Dem",
        "tuo":        "Dem",
        "se":         "Dem",
        "nämä":       "Dem",
        "nuo":        "Dem",
        "ne":         "Dem",

        # The relative "mikä" will be handled as a special case
        # separately so here we label all occurences of "mikä" as
        # interrogative.
        "mikä":       "Int",
        "kuka":       "Int",
        "ken":        "Int",  # ketä
        "kumpi":      "Int",
        "millainen":  "Int",
        "kuinka":     "Int",
        "miksi":      "Int",

        # The relative "joka" will be handled elsewhere. Here "joka"
        # is Voikko's lemmatization of jotakin, jollekin, jostakin, ...
        "joka":       "Ind",
        "kaikki":     "Ind",
        "jokainen":   "Ind",
        "koko":       "Ind",
        "harva":      "Ind",
        "muutama":    "Ind",
        "jokunen":    "Ind",
        "yksi":       "Ind",
        "ainoa":      "Ind",
        "eräs":       "Ind",
        "muuan":      "Ind",
        "joku":       "Ind",
        "jokin":      "Ind",
        "kukin":      "Ind",
        "moni":       "Ind",
        "usea":       "Ind",
        "molempi":    "Ind",
        "kumpikin":   "Ind",
        "kumpikaan":  "Ind",
        "jompikumpi": "Ind",
        "sama":       "Ind",
        "muu":        "Ind",
        "kukaan":     "Ind",
        "mikään":     "Ind",

        "toinen":     "Rcp"
    }
    pron_persons = {
        "minä":       "1",
        "sinä":       "2",
        "hän":        "3",
        "me":         "1",
        "te":         "2",
        "he":         "3"
    }
    infinite_moods = frozenset([
        "A-infinitive", "E-infinitive", "MA-infinitive", "MAINEN-infinitive"])

    def __init__(
            self,
            vocab: Vocab,
            name: str = "morphologizer",
            *,
            overwrite_lemma: bool = False,
    ) -> None:
        super().__init__()

        self.name = name
        self.vocab = vocab
        self.voikko = libvoikko.Voikko("fi")
        self.lookups = Lookups()
        self.overwrite_lemma = overwrite_lemma
        self.aux_labels = [vocab.strings.add(x) for x in ["aux", "aux:pass"]]
        self.cop_labels = [vocab.strings.add(x) for x in ["cop", "cop:own"]]
        self.nsubj_labels = [vocab.strings.add(x) for x in ["nsubj", "nsubj:cop"]]
        self.ccomp_labels = [vocab.strings.add(x) for x in ["csubj", "csubj:cop", "xcomp", "xcomp:ds"]]
        self.relcl_labels = [vocab.strings.add(x) for x in ["acl:relcl", "ccomp"]]
        self.foreign_tag = vocab.strings.add('Foreign')

    def __call__(self, doc: Doc) -> Doc:
        error_handler = self.get_error_handler()
        try:
            for token in doc:
                if token.pos in (PUNCT, SPACE):
                    if self.overwrite_lemma or token.lemma == 0:
                        token.lemma = token.orth
                else:
                    analysis = self._analyze(token)
                    morph = self.voikko_morph(token, analysis)
                    if morph:
                        token.set_morph(morph)
                    if self.overwrite_lemma or token.lemma == 0:
                        token.lemma_ = self.lemmatize(token, analysis)
            return doc
        except Exception as e:
            error_handler(self.name, self, [doc], e)

    def initialize(
        self,
        get_examples: Optional[Callable[[], Iterable[Example]]] = None,
        *,
        nlp: Optional[Language] = None,
        lookups: Optional[Lookups] = None,
    ):
        """Initialize the morphologizer and load in data.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        lookups (Lookups): The lookups object containing the (optional) tables
            such as "lemma_exc" and "morphologizer_exc". Defaults to None.
        """
        if lookups is None:
            lookups = load_lookups(lang=self.vocab.lang,
                                   tables=["lemma_exc", "morphologizer_exc"])
        self.lookups = lookups

    def voikko_morph(self, token: Token, analysis: dict) -> Optional[str]:
        # Run Voikko's analysis and convert the result to morph
        # features.
        exc_table = self.lookups.get_table("morphologizer_exc", {}).get(token.pos)
        if exc_table is not None:
            exc = exc_table.get(token.orth_.lower())
            if exc:
                return exc

        # Pre-compute some frequent morphs to avoid code duplication.
        # (Functions are not an option because the function call
        # overhead is too high.)

        # Clitic
        morph_clitic = None
        if "FOCUS" in analysis:
            focus = analysis["FOCUS"]
            if focus == "kin":
                morph_clitic = "Clitic=Kin"
            elif focus == "kaan":
                morph_clitic = "Clitic=Kaan"
            elif focus == "ka":
                morph_clitic = "Clitic=Ka"
        elif "KYSYMYSLIITE" in analysis:
            morph_clitic = "Clitic=Ko"

        morph_number = None
        morph_number_psor = None
        morph_person_psor = None
        if token.pos in (ADJ, ADP, ADV, AUX, NOUN, NUM, PRON, PROPN, VERB):
            # Number
            if "NUMBER" in analysis:
                number = analysis["NUMBER"]
                if number == "singular":
                    morph_number = "Number=Sing"
                elif number == "plural":
                    morph_number = "Number=Plur"

            # Number[psor] and Person[psor]
            if "POSSESSIVE" in analysis:
                possessive = analysis["POSSESSIVE"]
                if possessive == "1s":
                    morph_number_psor = "Number[psor]=Sing"
                    morph_person_psor = "Person[psor]=1"
                elif possessive == "1p":
                    morph_number_psor = "Number[psor]=Plur"
                    morph_person_psor = "Person[psor]=1"
                elif possessive == "3":
                    morph_person_psor = "Person[psor]=3"

        # Set morphs per POS
        morphology = []
        if token.pos in (ADJ, NOUN, PROPN):
            # Abbr
            if "CLASS" in analysis and analysis["CLASS"] == "lyhenne":
                morphology.append("Abbr=Yes")

            # Case
            if "SIJAMUOTO" in analysis:
                morphology.append(self.voikko_cases[analysis["SIJAMUOTO"]])

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)

            # Degree
            if token.pos == ADJ and "COMPARISON" in analysis:
                morphology.append(self.voikko_degree[analysis["COMPARISON"]])

            # Number
            if morph_number is not None:
                morphology.append(morph_number)

            # Number[psor]
            if morph_number_psor is not None:
                morphology.append(morph_number_psor)

            # NumType
            if token.pos == ADJ and "NUMTYPE" in analysis:
                morphology.append(f'NumType={analysis["NUMTYPE"]}')

            # Person[psor]
            if morph_person_psor is not None:
                morphology.append(morph_person_psor)

        elif token.pos in (AUX, VERB):
            vclass = analysis.get("CLASS")

            # Abbr
            if vclass == "lyhenne":
                morphology.append("Abbr=Yes")

            # Case
            if "SIJAMUOTO" in analysis:
                morphology.append(self.voikko_cases[analysis["SIJAMUOTO"]])

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)
            
            # Connegative
            if "CONNEGATIVE" in analysis:
                morphology.append("Connegative=Yes")

            # Degree
            if "COMPARISON" in analysis:
                morphology.append(self.voikko_degree[analysis["COMPARISON"]])

            # InfForm and Mood
            # These are mutually exclusive and both are based on MOOD
            mood = None
            if "MOOD" in analysis:
                mood = analysis["MOOD"]
                morph_inf_form_or_mood = self.voikko_mood.get(mood)
                if morph_inf_form_or_mood is not None:
                    morphology.append(morph_inf_form_or_mood)

            # Number
            if morph_number is not None:
                morphology.append(morph_number)

            # Number[psor]
            if morph_number_psor is not None:
                morphology.append(morph_number_psor)

            # PartForm
            participle = None
            if "PARTICIPLE" in analysis:
                participle = analysis["PARTICIPLE"]
                morph_part_form = self.voikko_part_form.get(participle)
                if morph_part_form:
                    morphology.append(morph_part_form)

            # Person
            person = None
            if "PERSON" in analysis:
                person = analysis["PERSON"]
                if person in ("0", "1", "2", "3"):
                    morphology.append(f"Person={person}")

            # Person[psor]
            if morph_person_psor is not None:
                morphology.append(morph_person_psor)

            # Polarity
            if vclass == "kieltosana":
                morphology.append("Polarity=Neg")

            # Tense
            if "TENSE" in analysis:
                morphology.append(self.voikko_tense[analysis["TENSE"]])

            # VerbForm
            if mood in self.infinite_moods:
                morphology.append("VerbForm=Inf")
            elif participle is not None:
                morphology.append("VerbForm=Part")
            else:
                morphology.append("VerbForm=Fin")

            # Voice
            if person in ("0", "1", "2", "3"):
                morphology.append("Voice=Act")
            elif person == "4":
                morphology.append("Voice=Pass")
            elif "VOICE" in analysis:
                morphology.append(f"Voice={analysis['VOICE']}")
            elif participle == "past_passive":
                morphology.append("Voice=Pass")
            elif participle in ("present_active", "past_active", "present_passive"):
                morphology.append("Voice=Act")

        elif token.pos == ADV:
            # Abbr
            if "CLASS" in analysis and analysis["CLASS"] == "lyhenne":
                morphology.append("Abbr=Yes")

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)

            # Degree
            if "COMPARISON" in analysis:
                degree = analysis["COMPARISON"]
                if degree in ("comparative", "superlative"):
                    morphology.append(self.voikko_degree[degree])

            # Number[psor]
            if morph_number_psor is not None:
                morphology.append(morph_number_psor)

            # Person[psor]
            if morph_person_psor is not None:
                morphology.append(morph_person_psor)

        elif token.pos == PRON:
            # Case
            if "SIJAMUOTO" in analysis:
                morphology.append(self.voikko_cases[analysis["SIJAMUOTO"]])

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)

            # Degree
            if "COMPARISON" in analysis:
                morphology.append(self.voikko_degree[analysis["COMPARISON"]])
            
            # Number
            if morph_number is not None:
                morphology.append(morph_number)

            # Number[psor]
            if morph_number_psor is not None:
                morphology.append(morph_number_psor)

            # Person
            if "PERSON" in analysis:
                person = analysis["PERSON"]
                if person in ("0", "1", "2", "3"):
                    morphology.append(f"Person={person}")

            # Person[psor]
            if morph_person_psor is not None:
                morphology.append(morph_person_psor)

            # PronType
            if "PRONTYPE" in analysis:
                morphology.append(f"PronType={analysis['PRONTYPE']}")

            # Reflex
            if "BASEFORM" in analysis and analysis["BASEFORM"] == "itse":
                morphology.append("Reflex=Yes")

        elif token.pos in (CCONJ, SCONJ):
            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)

        elif token.pos == NUM:
            # Abbr
            if "CLASS" in analysis and analysis["CLASS"] == "lyhenne":
                morphology.append("Abbr=Yes")

            # Case
            if "SIJAMUOTO" in analysis:
                morphology.append(self.voikko_cases[analysis["SIJAMUOTO"]])

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)
                
            # Number
            if morph_number is not None:
                morphology.append(morph_number)

            # NumType
            if "NUMTYPE" in analysis:
                morphology.append(f'NumType={analysis["NUMTYPE"]}')

        elif token.pos == ADP:
            # AdpType
            if "ADPTYPE" in analysis:
                morphology.append(f"AdpType={analysis['ADPTYPE']}")

            # Clitic
            if morph_clitic is not None:
                morphology.append(morph_clitic)

            # Number[psor]
            if morph_number_psor is not None:
                morphology.append(morph_number_psor)

            # Person[psor]
            if morph_person_psor is not None:
                morphology.append(morph_person_psor)

        elif token.pos == SYM:
            # Case
            if "SIJAMUOTO" in analysis:
                morphology.append(self.voikko_cases[analysis["SIJAMUOTO"]])
            
        elif token.tag == self.foreign_tag:
            # Foreign
            morphology.append('Foreign=Yes')

        return "|".join(morphology) if morphology else None

    def lemmatize(self, token: Token, analysis: dict) -> str:
        cached_lower = None
        exc_table = self.lookups.get_table("lemma_exc", {}).get(token.pos)
        if exc_table is not None:
            cached_lower = token.orth_.lower()
            exc = exc_table.get(cached_lower)
            if exc:
                return exc

        # Some exceptions to Voikko's lemmatization algorithm to
        # better match UD lemmas
        if token.pos in (AUX, VERB) and "PARTICIPLE" in analysis:
            return self._participle_lemma(analysis)
        elif token.pos == NOUN and analysis.get("MOOD") == "MINEN-infinitive":
            return self._minen_noun_lemma(analysis)
        elif token.pos in (NOUN, NUM, PROPN) and (colon_i := token.orth_.find(":")) > 0:
            # Lemma of inflected abbreviations: BBC:n, EU:ssa
            return token.orth_[:colon_i]
        elif token.pos == ADV:
            cached_lower = cached_lower or token.orth_.lower()
            return self._adv_lemma(analysis, cached_lower)
        elif token.pos == ADP:
            return cached_lower or token.orth_.lower()
        elif not "BASEFORM" in analysis:
            if token.pos in (PROPN, INTJ, SYM, X):
                return token.orth_
            else:
                return cached_lower or token.orth_.lower()
        else:
            return analysis["BASEFORM"]

    def _analyze(self, token):
        orth = token.orth_
        if '-' in orth:
            # Analyze only the head token on hyphenated compound
            # words.
            parts = [x for x in orth.rsplit('-', 1) if x]
            if parts:
                orth = parts[-1]
        else:
            parts = [orth]

        analyses = self.voikko.analyze(orth)
        analysis = self._disambiguate_analyses(token, analyses)
        if len(parts) > 1 and "BASEFORM" in analysis:
            analysis["BASEFORM"] = parts[0] + "-" + analysis["BASEFORM"]
        return self._enrich_voikko_analysis(token, analysis)

    def _enrich_voikko_analysis(self, token, analysis):
        # Enrich Voikko's analysis with extra features

        if token.pos in (AUX, VERB):
            # connegative
            if analysis.get("TENSE") == "present_simple" and (
                    # "en [ole]", "et [halua]", "ei [maksaisi]"
                    self._last_aux_is_negative(token.lefts)

                    or

                    # "et [ole] nähnyt", "emme [ehdi] tutustua"
                    (token.dep in self.aux_labels and
                     self._last_aux_is_negative(t for t in token.head.lefts if t.i < token.i))

                    or

                    # "minulla ei [ole] autoa", "tietoja ei [ole] saatu"
                    (token.dep in self.cop_labels and
                     self._last_aux_is_negative(t for t in token.head.children if t.i < token.i))
            ):
                analysis["CONNEGATIVE"] = True
                if analysis.get("MOOD") == "imperative":
                    analysis["MOOD"] = "indicative"
                if "PERSON" in analysis and analysis["PERSON"] != "4":
                    del analysis["PERSON"]
                if "NUMBER" in analysis:
                    del analysis["NUMBER"]

            # correlated voice in verb chains
            if "PERSON" not in analysis and "CONNEGATIVE" not in analysis:
                corr_person = None
                auxs = [t for t in token.lefts if t.dep == aux]
                if auxs:
                    # voimme [haluta]
                    corr_person = auxs[-1].morph.get("Person")
                elif token.head.pos == VERB:
                    # lähdimme [kävelemään]
                    corr_person = token.head.morph.get("Person")

                if corr_person:
                    person = corr_person[0]
                    if person in ("1", "2", "3"):
                        analysis["VOICE"] = "Act"
                    elif person == "4":
                        analysis["VOICE"] = "Pass"

            # "eikä" has the clitic "ka"
            if analysis.get("CLASS") == "kieltosana":
                if token.orth_.lower().endswith("kä"):
                    analysis["FOCUS"] = "ka"

                # UD doesn't assign a mood for the negative verb
                if "MOOD" in analysis:
                    del analysis["MOOD"]

        if token.pos in (NUM, ADJ):
            if analysis.get("CLASS") == "lukusana":
                # NumType
                base = analysis.get("BASEFORM", "")
                if (base.endswith(".") or
                    base.endswith("s") or
                    base.endswith("stoista") or
                    base in ["ensimmäinen", "toinen"] or
                    # A hack for recognizing ordinal numbers("1.", "2.",
                    # "3.", ...) until the tokenizer is fixed.
                    (base.isdigit() and token.i < len(token.doc) - 1 and token.nbor(1).orth_ == ".")
                ):
                    analysis["NUMTYPE"] = "Ord"
                elif base:
                    analysis["NUMTYPE"] = "Card"

        elif token.pos == ADP:
            # adpositions: pre- or postposition?
            if token.head.i < token.i:
                analysis["ADPTYPE"] = "Post"
            else:
                analysis["ADPTYPE"] = "Prep"

            if "NUMBER" in analysis:
                del analysis["NUMBER"]

        elif token.pos == PRON:
            # Pronoun types
            base = analysis.get("BASEFORM")
            if self._is_relative_pronoun(token, base):
                analysis["PRONTYPE"] = "Rel"
            else:
                pron_type = self.pron_types.get(base)
                if pron_type:
                    analysis["PRONTYPE"] = pron_type
                if pron_type == "Prs":
                    person = self.pron_persons.get(base)
                    if person:
                        analysis["PERSON"] = person

        elif token.pos == PROPN:
            # Detecting plural suffix. Might not work properly on
            # foreign names...
            if "NUMBER" not in analysis and not token.orth_.endswith("t"):
                analysis["NUMBER"] = "singular"

        elif token.pos in (ADV, CCONJ, SCONJ, SYM, INTJ, X):
            # Cleanup extra features not in UD
            if "NUMBER" in analysis:
                del analysis["NUMBER"]

        # Abbreviation cases: YK:n, BKT:stä
        if token.pos in (NOUN, NUM, PROPN) and "SIJAMUOTO" not in analysis:
            i = token.orth_.find(":")
            if i > 0:
                affix = token.orth_[(i+1):]
                sijamuoto = self.affix_to_sijamuoto.get(affix)
                if sijamuoto:
                    analysis["SIJAMUOTO"] = sijamuoto

        return analysis

    def _disambiguate_analyses(self, token, analyses):
        matching_pos = [x for x in analyses if self._analysis_has_compatible_pos(token, x)]
        if matching_pos:
            analyses = matching_pos

        if len(analyses) > 1:
            # Disambiguate among multiple possible analyses

            if token.pos == VERB:
                if token.dep == conj:
                    coord_head = token.head
                else:
                    coord_head = token
                negative = any(
                    t.dep == aux and "Neg" in t.morph.get("Polarity")
                    for t in coord_head.lefts)
                if negative:
                    analyses = [x for x in analyses if x.get("NEGATIVE") != "false"] or analyses
                else:
                    analyses = [x for x in analyses if x.get("NEGATIVE") != "true"] or analyses

                if (any(t.dep == aux or t.dep == cop for t in coord_head.lefts) and
                    any(x.get("PARTICIPLE") == "past_active" for x in analyses)):
                    # "olen haaveillut"
                    analyses = self._prefer_active(analyses)
                elif token.dep in self.ccomp_labels or \
                     token.dep == acl or \
                     any(t.dep == aux and "Neg" not in t.morph.get("Polarity") for t in coord_head.lefts):
                    # verbiketjut: "saa [nähdä]", "osaat [uida]", "voi [veistää]"
                    analyses = self._prefer_infinite_form(analyses)
                elif token.dep == conj and token.head.morph.get("InfForm"):
                    # If we are coordinated with an infinite form we
                    # should also have the same (TODO) infinitive form.
                    #
                    # "tarkoitus on tutkia ja todistaa"
                    analyses = self._prefer_infinite_form(analyses)
                else:
                    analyses = self._prefer_indicative_form(analyses)

            elif token.pos == AUX:
                negative = any(
                    t.dep == aux and "Neg" in t.morph.get("Polarity")
                    for t in token.head.lefts)
                if negative:
                    analyses = [x for x in analyses if x.get("NEGATIVE") != "false"] or analyses
                else:
                    analyses = [x for x in analyses if x.get("NEGATIVE") != "true"] or analyses

                if any(x.get("PARTICIPLE") == "past_active" for x in analyses):
                    # "olen ollut", "ei ollut käyty"
                    analyses = self._prefer_active(analyses)

            elif token.pos == NUM:
                # For numbers like 1,5 prefer the analysis without
                # SIJAMUOTO and NUMBER because UD doesn't have them,
                # either.
                analyses = [x for x in analyses if "SIJAMUOTO" not in x] or analyses

                # Prefer uppercase Roman numerals
                if all(self.roman_numeral_structure_re.fullmatch(x.get("STRUCTURE", ""))
                       for x in analyses):
                    analyses = [x for x in analyses if "j" in x.get("STRUCTURE")]

            elif token.pos in (NOUN, PRON) and \
                 ((token.dep in self.nsubj_labels) or \
                  (token.dep == conj and token.head.dep in self.nsubj_labels)):
                # Subject is usually nominative, genetive or partitive
                analyses = [
                    x for x in analyses
                    if x.get("SIJAMUOTO") in ["nimento", "omanto", "osanto"]
                ] or analyses

            elif token.pos in (NOUN, PRON) and \
                 (token.dep == obj or (token.dep == conj and token.head.dep == obj)):
                # Object is usually partitive, accusative or genetive
                analyses = [
                    x for x in analyses
                    if x.get("SIJAMUOTO") in ("kohdanto", "omanto", "osanto")
                ] or analyses

            elif token.pos == NOUN:
                # Prefer non-compound words.
                # e.g. "asemassa" will be lemmatized as "asema", not "ase#massa"
                analyses = sorted(analyses, key=self._is_compound_word)

            elif token.pos == ADJ:
                # Prefer laatusana over nimisana_laatusana
                analyses = [x for x in analyses if x.get("CLASS") == "laatusana"] or analyses

        if analyses:
            return analyses[0]
        else:
            return {}

    def _analysis_has_compatible_pos(self, token, analysis):
        if "CLASS" not in analysis:
            return False

        tpos = token.pos
        vclass = analysis["CLASS"]

        return (
            vclass in self.voikko_classes_by_pos[tpos]

            or

            (tpos == NOUN and vclass == "teonsana" and analysis.get("MOOD") == "MINEN-infinitive")

            or

            (tpos == VERB and (
                (vclass == "teonsana" and analysis.get("MOOD") != "MINEN-infinitive") or
                ("PARTICIPLE" in analysis and (
                    # agent participle
                    (vclass == "nimisana" and analysis["PARTICIPLE"] == "agent") or
                    # VA, NUT and TU participles
                    (vclass == "laatusana" and analysis["PARTICIPLE"] in (
                        "past_active", "past_passive", "present_active", "present_passive"))))))

            or

            (tpos == ADV and
             vclass in ("laatusana", "lukusana") and
             analysis.get("SIJAMUOTO") == "kerrontosti")
        )

    def _prefer_infinite_form(self, analyses):
        infinite = [x for x in analyses if x.get("MOOD") in self.infinite_moods]
        return infinite or analyses

    def _prefer_indicative_form(self, analyses):
        indicative = [x for x in analyses if x.get("MOOD") == "indicative"]
        return indicative or analyses

    def _prefer_active(self, analyses):
        active = [x for x in analyses if x.get("PARTICIPLE") == "past_active"]
        return active or analyses

    def _minen_noun_lemma(self, analysis):
        fstoutput = analysis.get("FSTOUTPUT")
        ny_match = self.ny_re.search(fstoutput)
        if ny_match:
            return ny_match.group(1)

        fst_match = self.minen_re.search(fstoutput)
        if not fst_match:
            return None

        stem = fst_match.group(1)
        compounds = self.compound_re.findall(analysis.get("WORDBASES"))
        if len(compounds) > 1:
            return "".join(compounds[:-1]) + stem + "minen"
        else:
            return stem + "minen"

    def _is_relative_pronoun(self, token, baseform):
        if baseform in ("joka", "kuka", "mikä"):
            relcl_head = self._relative_clause_head(token)
            if relcl_head is not None:
                i = relcl_head.left_edge.i
                return ((i == token.i) or
                        ((i + 1 == token.i) and token.nbor(-1).orth_ == ",") or
                        (token.i > 0 and token.nbor(-1).dep == cc))

        # FIXME: independent relative clauses: "Maksan mitä pyydät."

        return False

    def _relative_clause_head(self, token):
        t = token
        while t.head != t:
            if t.dep in self.relcl_labels:
                return t
            t = t.head
        return None

    def _last_aux_is_negative(self, tokens):
        auxs = [t for t in tokens if t.dep in self.aux_labels]
        return bool(auxs and ("Neg" in auxs[-1].morph.get("Polarity")))

    def _participle_lemma(self, analysis):
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

    def _adv_lemma(self, analysis, word):
        focus = analysis.get("FOCUS")
        has_kysymysliite = "KYSYMYSLIITE" in analysis

        if focus and has_kysymysliite:
            k = 2
        elif focus or has_kysymysliite:
            k = 1
        else:
            k = 0
        for _ in range(k):
            if focus and word.endswith(focus):
                word = word[:-len(focus)]
            elif has_kysymysliite and (word.endswith("ko") or word.endswith("kö")):
                word = word[:-2]

        if "POSSESSIVE" in analysis and analysis["POSSESSIVE"] != "3":
            word = self._remove_possessive_suffix(word, analysis)

        return word

    def _remove_possessive_suffix(self, word, analysis):
        """Removes possessive suffix from the word.

        Example: "kanssamme" -> "kanssa"
        """
        suffixes = self.possessive_suffixes[analysis["POSSESSIVE"]]
        suffix = next((s for s in suffixes if word.endswith(s)))
        if not suffix:
            return word

        word = word[:-len(suffix)]
        if analysis.get("SIJAMUOTO") == "tulento" and word.endswith("e"):
            # onne+kse+mme -> onne+ksi
            word = word[:-1] + "i"
        elif analysis.get("SIJAMUOTO") == "sisatulento":
            # lapsee+ni -> lapseen
            word = word + "n"
        elif suffix.startswith('n') and analysis.get("BASEFORM").endswith("n"):
            # mukaa+ni -> mukaan
            word = word + "n"

        return word

    def _is_compound_word(self, analysis):
        structure = analysis.get('STRUCTURE', '')
        return structure.count('=') > 1

    def score(self, examples, **kwargs):
        def morph_key_getter(token, attr):
            return getattr(token, attr).key

        validate_examples(examples, "MorphologizerLemmatizer.score")
        results = {}
        results.update(Scorer.score_token_attr(examples, "morph", getter=morph_key_getter, **kwargs))
        results.update(Scorer.score_token_attr_per_feat(examples,
            "morph", getter=morph_key_getter, **kwargs))
        results.update(Scorer.score_token_attr(examples, "lemma", **kwargs))
        return results

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ):
        serialize = {"lookups": lambda p: self.lookups.to_disk(p)}
        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "MorphologizerLemmatizer":
        deserialize = {"lookups": lambda p: self.lookups.from_disk(p)}
        util.from_disk(path, deserialize, exclude)
        return self

    def to_bytes(self, *, exclude: Iterable[str] = SimpleFrozenList()) -> bytes:
        serialize = {"lookups": self.lookups.to_bytes}
        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "MorphologizerLemmatizer":
        deserialize = {"lookups": lambda b: self.lookups.from_bytes(b)}
        util.from_bytes(bytes_data, deserialize, exclude)
        return self


@Finnish.factory(
    "morphologizer_lemmatizer",
    assigns=["token.morph", "token.lemma"],
    requires=["token.pos", "token.dep"],
    default_score_weights={"morph_acc": 1.0, "morph_per_feat": None, "lemma_acc": 0.0},
)
def make_morphologizer_lemmatizer(
    nlp: Language,
    name: str,
    overwrite_lemma: bool = False
):
    return MorphologizerLemmatizer(nlp.vocab, name, overwrite_lemma=overwrite_lemma)


class VrtZipCorpus:
    """Iterate Doc objects from a ZIP file that contains VRT files.

    path (Path): The ZIP filename to read from.
    min_length (int): Minimum document length (in tokens). Shorter documents
        will be skipped. Defaults to 0, which indicates no limit.
    max_length (int): Maximum document length (in tokens). Longer documents will
        be skipped. Defaults to 0, which indicates no limit.
    limit (int): Limit corpus to a subset of examples, e.g. for debugging.
        Defaults to 0, which indicates no limit.
    """

    def __init__(
            self,
            path: Union[str, Path],
            *,
            limit: int = 0,
            min_length: int = 0,
            max_length: int = 0,
    ) -> None:
        self.path = util.ensure_path(path)
        self.limit = limit
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, nlp: "Language") -> Iterator[Example]:
        """Yield examples from the data.

        nlp (Language): The current nlp object.
        YIELDS (Example): The example objects.

        DOCS: https://spacy.io/api/corpus#jsonlcorpus-call
        """
        i = 0
        zf = ZipFile(self.path)
        vrt_names = [p for p in zf.namelist() if p.endswith('.VRT')]
        for nested in vrt_names:
            with zf.open(nested) as f:
                ftext = TextIOWrapper(f, encoding='utf-8')
                for text in self.vrt_extract_documents(ftext):
                    text = self.skip_title_line(text)
                    doc = nlp.make_doc(text)
                    if self.min_length >= 1 and len(doc) < self.min_length:
                        continue
                    elif self.max_length >= 1 and len(doc) >= self.max_length:
                        continue
                    else:
                        words = [w.text for w in doc]
                        spaces = [bool(w.whitespace_) for w in doc]
                        # We don't *need* an example here, but it seems nice to
                        # make it match the Corpus signature.
                        yield Example(doc, Doc(nlp.vocab, words=words, spaces=spaces))

                    i += 1
                    if self.limit >= 1 and i >= self.limit:
                        return


    def vrt_extract_documents(self, fileobj):
        tokens = []
        quote_active = False
        paragraph_break = False
        for line in fileobj:
            if line.startswith('</doc'):
                # end of document
                yield ''.join(tokens)

                tokens = []
                quote_active = False
                paragraph_break = False

            elif line.startswith('</paragraph'):
                # paragraph break
                quote_active = False
                paragraph_break = True

            elif line.startswith('<'):
                # ignored document structure
                pass

            else:
                # content
                fields = line.split('\t')
                term = fields[1]

                if not tokens:
                    pass
                elif paragraph_break:
                    tokens.append('\n\n')
                elif not (self.char_is_in(term, '"”') and quote_active) \
                     and not (term == '’') \
                     and not (term.isdigit() and len(tokens) >= 2 and tokens[-2].isdigit() and self.char_is_in(tokens[-1], '.,')) \
                     and not self.char_is_in(term, '.,:;)]}') \
                     and not self.char_is_in(tokens[-1], '([{’') \
                     and not (self.char_is_in(tokens[-1], '"”') and quote_active):
                    tokens.append(' ')

                tokens.append(term)

                paragraph_break = False
                if self.char_is_in(term, '"”'):
                    quote_active = not quote_active
                elif not quote_active and len(term) > 1 and self.char_is_in(term[0], '"”'):
                    # Sometimes the starting quote is not tokenized as a
                    # separate token
                    quote_active = True

        if tokens:
            # We end up here if the last document wasn't terminated
            # properly with </doc>
            yield ''.join(tokens)

    def skip_title_line(self, text):
        # The first line is the title, the second line is empty
        parts = text.split('\n', 2)
        if len(parts) < 3:
            return text
        else:
            assert len(parts[1]) == 0
            return parts[-1]

    def char_is_in(self, x, chars):
        return len(x) == 1 and any(x == c for c in chars)


def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Span]:
    """Detect base noun phrases from a dependency parse. Works on both Doc and Span."""
    labels = [
        "nsubj",
        "nsubj:cop",
        "obj",
        "obl",
        "ROOT",
    ]
    extend_labels = [
        "advmod",
        "amod",
        "appos",
        "case",
        "compound",
        "compound:nn",
        "flat:name",
        "nmod",
        "nmod:gobj",
        "nmod:gsubj",
        "nmod:poss",
        "nummod",
    ]

    def potential_np_head(word):
        # TODO: PRON handling is inconsistent. Should some pronouns
        # (indefinite?, personal?) be considered part of a noun chunk?
        return word.pos in (NOUN, PROPN) and (word.dep in np_deps or word.head.pos == PRON)

    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    np_deps = [doc.vocab.strings[label] for label in labels]
    extend_deps = [doc.vocab.strings[label] for label in extend_labels]
    np_label = doc.vocab.strings.add("NP")
    conj_label = doc.vocab.strings.add("conj")

    rbracket = 0
    prev_end = -1
    for i, word in enumerate(doclike):
        if i < rbracket:
            continue

        # Is this a potential independent NP head or coordinated with
        # a NOUN that is itself an independent NP head?
        #
        # e.g. "Terveyden ja hyvinvoinnin laitos"
        if potential_np_head(word) or (word.dep == conj_label and potential_np_head(word.head)):
            # Try to extend to the left to include adjective/num
            # modifiers, compound words etc.
            lbracket = word.i
            for ldep in word.lefts:
                if ldep.pos in (NOUN, PROPN, NUM, ADJ) and ldep.dep in extend_deps:
                    lbracket = ldep.left_edge.i
                    break

            # Prevent nested chunks from being produced
            if lbracket <= prev_end:
                continue

            rbracket = word.i
            # Try to extend the span to the right to capture close
            # appositions and noun modifiers
            for rdep in word.rights:
                if rdep.dep in extend_deps:
                    rbracket = rdep.i
                    for j in range(rdep.i + 1, rdep.right_edge.i + 1):
                        if doc[j].dep in extend_deps:
                            rbracket = j
            prev_end = rbracket

            yield lbracket, rbracket + 1, np_label


@util.registry.misc("spacyfi.read_lookups_from_json.v1")
def create_lookups_from_json_reader(path: Path) -> Lookups:
    lookups = Lookups()
    for p in path.glob("*.json"):
        table_name = p.stem
        data = srsly.read_json(p)
        lookups.add_table(table_name, data)
    return lookups


@util.registry.readers("spacyfi.VrtZipCorpus.v1")
def create_vrt_zip_reader(
        path: Optional[Path], min_length: int = 0, max_length: int = 0, limit: int = 0
) -> Callable[["Language"], Iterable[Doc]]:
    if path is None:
        raise ValueError(Errors.E913)

    return VrtZipCorpus(
        path,
        limit=limit,
        min_length=min_length,
        max_length=max_length,
    )


class FinnishExDefaults(FinnishDefaults):
    syntax_iterators = {"noun_chunks": noun_chunks}


@util.registry.languages("fi")
class FinnishExtended(Language):
    """Extends the default Finnish language class with syntax iterators."""
    lang = 'fi'
    Defaults = FinnishExDefaults
