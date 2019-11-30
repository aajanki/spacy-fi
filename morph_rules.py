from itertools import chain
from spacy.symbols import LEMMA


def morph_rule(lemma, pron_type, number, case, person=None):
    morph = {
        LEMMA: lemma,
        "POS": "PRON",
        "PronType": pron_type,
        "Number": number,
        "Case": case,
    }
    if person is not None:
        morph["person"] = person
    return morph


def build_morphs(pron_type, forms, lemmas, numbers, persons=None):
    if persons is None:
        expanded_zips = (
            zip(words, lemmas, [pron_type]*len(words), numbers, [case]*len(words))
            for case, words in forms.items()
        )
    else:
        expanded_zips = (
            zip(words, lemmas, [pron_type]*len(words), numbers, [case]*len(words), persons)
            for case, words in forms.items()
        )

    expanded = chain.from_iterable(expanded_zips)
    expanded = (x for x in expanded if x[0] is not None)
    return {
        word: morph_rule(*rest) for (word, *rest) in expanded
    }


# Person pronouns: http://scripta.kotus.fi/visk/sisallys.php?p=100
prs_pron = {
    "Nom": ["minä", "mä", "sinä", "sä", "hän", "me", "te", "he"],
    "Gen": ["minun", "mun", "sinun", "sun", "hänen", "meidän", "teidän", "heidän"],
    "Acc": ["minut", "mut", "sinut", "sut", "hänet", "meidät", "teidät", "heidät"],
    "Par": ["minua", "mua", "sinua", "sua", "häntä", "meitä", "teitä", "heitä"],
    "Ess": ["minuna", None, "sinuna", None, "hänenä", "meinä", "teinä", "heinä"],
    "Tra": ["minuksi", None, "sinuksi", None, "häneksi", "meiksi", "teiksi", "heiksi"],
    "Ine": ["minussa", "mussa", "sinussa", "sussa", "hänessä", "meissä", "teissä", "heissä"],
    "Ela": ["minusta", "musta", "sinusta", "susta", "hänestä", "meistä", "teistä", "heistä"],
    "Ill": ["minuun", "muhun", "sinuun", "suhun", "häneen", "meihin", "teihin", "heihin"],
    "Ade": ["minulla", "mulla", "sinulla", "sulla", "hänellä", "meillä", "teillä", "heillä"],
    "Abl": ["minulta", "multa", "sinulta", "sulta", "häneltä", "meiltä", "teiltä", "heiltä"],
    "All": ["minulle", "mulle", "sinulle", "sulle", "hänelle", "meille", "teille", "heille"]
}
prs_persons = ["One", "One", "Two", "Two", "Three", "One", "Two", "Three"]
prs_numbers = ["Sing", "Sing", "Sing", "Sing", "Sing", "Plur", "Plur", "Plur"]
prs_lemmas = ["minä", "minä", "sinä", "sinä", "hän", "minä", "sinä", "hän"]


# Demonstrative pronouns, http://scripta.kotus.fi/visk/sisallys.php?p=101
dem_pron = {
    "Nom": ["se", "ne", "tämä", "nämä", "tuo", "nuo"],
    "Gen": ["sen", "niiden", "tämän", "näiden", "tuon", "noiden"],
    "Par": ["sitä", "niitä", "tätä", "näitä", "tuota", "noita"],
    "Ess": ["sinä", "niinä", "tänä", "näinä", "tuona", "noina"],
    "Tra": ["siksi", "niiksi", "täksi", "näiksi", "tuoksi", "noiksi"],
    "Ine": ["siinä", "niissä", "tässä", "näissä", "tuossa", "noissa"],
    "Ela": ["siitä", "niistä", "tästä", "näistä", "tuosta", "noista"],
    "Ill": ["siihen", "niihin", "tähän", "näihin", "tuohon", "noihin"],
    "Ade": ["sillä", "niillä", "tällä", "näillä", "tuolla", "noilla"],
    "Abl": ["siltä", "niiltä", "tältä", "näiltä", "tuolta", "noilta"],
    "All": ["sille", "niille", "tälle", "näille", "tuolle", "noille"],
    "Com": [None, "niine", None, "näine", None, "noine"]
}
dem_numbers = ["Sing", "Sing", "Sing", "Plur", "Plur", "Plur"]
dem_lemmas = ["se", "se", "tämä", "tämä", "tuo", "tuo"]


# Relative pronouns, http://scripta.kotus.fi/visk/sisallys.php?p=102
rel_pron = {
    "Nom": ["joka", "jotka", "mikä", "mitkä"],
    "Gen": ["jonka", "joiden", "minkä", None],
    "Par": ["jota", "joita", "mitä", None],
    # Essive case of "mikä" is "minä". It's skipped because it would
    # override the singular first-person pronoun
    "Ess": ["jona", "joina", None, None],
    "Tra": ["joksi", "joiksi", "miksi", None],
    "Ine": ["jossa", "joissa", "missä", None],
    "Ela": ["josta", "joista", "mistä", None],
    "Ill": ["johon", "joihin", "mihin", None],
    "Ade": ["jolla", "joilla", "millä", None],
    "Abl": ["jolta", "joilta", "miltä", None],
    "All": ["jolle", "joille", "mille", None],
}
rel_numbers = ["Sing", "Plur", "Sing", "Plur"]
rel_lemmas = ["joka", "joka", "mikä", "mikä"]


# Interrogative pronouns, http://scripta.kotus.fi/visk/sisallys.php?p=102
int_pron = {
    "Nom": ["kuka", "ketkä", "kumpi"],
    "Gen": ["kenen", "keiden", "kumman"],
    "Acc": ["kenet", None, None],
    "Par": ["ketä", "keitä", "kumpaa"],
    "Ess": ["kenenä", "keinä", "kumpana"],
    "Tra": ["keneksi", "keiksi", "kummaksi"],
    "Ine": ["kenessä", "keissä", "kummassa"],
    "Ela": ["kenestä", "keistä", "kummasta"],
    "Ill": ["kehen", "keihin", "kumpaan"],
    "Ade": ["kenellä", "keillä", "kummalla"],
    "Abl": ["keneltä", "keiltä", "kummalta"],
    "All": ["kenelle", "keille", "kummalle"],
}
int_numbers = ["Sing", "Plur", "Sing"]
int_lemmas = ["kuka", "kuka", "kumpi"]


# quantifier pronouns, http://scripta.kotus.fi/visk/sisallys.php?p=103
ind_pron = {
    "Nom": ["joku", "jotkut", "jokin", "jotkin", "muutama"],
    "Gen": ["jonkun", "joidenkuiden", "jonkin", "joidenkin", "muutaman"],
    "Par": ["jotakuta", "joitakuita", "jotain", "joitain", "muutamaa"],
    "Ess": ["jonakuna", "joinakuina", "jonain", "joinain", "muutamana"],
    "Tra": ["joksikuksi", "joiksikuiksi", "joksikin", "joiksikin", "muutamaksi"],
    "Ine": ["jossakussa", "joissakuissa", "jossain", "joissain", "muutamassa"],
    "Ela": ["jostakusta", "joistakuista", "jostain", "joistain", "muutamasta"],
    "Ill": ["johonkuhun", "joihinkuihin", "johonkin", "joihinkin", "muutamaan"],
    "Ade": ["jollakulla", "joillakuilla", "jollain", "joillain", "muutamalla"],
    "Abl": ["joltakulta", "joiltakuilta", "joltain", "joiltain", "muutamalta"],
    "All": ["jollekulle", "joillekuille", "jollekin", "joillekin", "muutamalle"],
}
ind_numbers = ["Sing", "Plur", "Sing", "Plur", "Sing"]
ind_lemmas = ["joku", "joku", "jokin", "jokin", "muutama"]


ind2_pron = {
    "Nom": ["kukaan", "ketkään", "mikään", "jokainen", "kaikki"],
    "Gen": ["kenenkään", "keidenkään", "minkään", "jokaisen", "kaiken"],
    "Par": ["ketään", "keitään", "mitään", "jokaista", "kaikkea"],
    "Ess": ["kenään", None, "minään", "jokaisena" "kaikkena"],
    "Tra": ["keneksikään", None, "miksikään", "jokaiseksi", "kaikeksi"],
    "Ine": ["kenessäkään", "keissäkään", "missään", "jokaisessa", "kaikissa"],
    "Ela": ["kenestäkään", "keistäkään", "mistään", "jokaisesta", "kaikista"],
    "Ill": ["kehenkään", "keihinkään", "mihinkään", "jokaiseen", "kaikkeen"],
    "Ade": ["kenelläkään", "keilläkään", "millään", "jokaisella", "kaikilla"],
    "Abl": ["keneltäkään", "keiltäkään", "miltään", "jokaiselta", "kaikilta"],
    "All": ["kenellekään", "keillekään", "millekään", "jokaiselle", "kaikille"]
}
ind2_numbers = ["Sing", "Plur", "Sing", "Sing", "Plur"]
ind2_lemmas = ["kukaan", "kukaan", "mikään", "jokainen", "kaikki"]


ind3_pron = {
    "Nom": ["muu", "muut", "moni", "monet", "usea", "useat"],
    "Gen": ["muun", "muiden", "monen", "monien", "usean", "useiden"],
    "Par": ["muuta", "muita", "monta", "monia", "useaa", "useita"],
    "Ess": ["muuna", "muina", "monena", "monina", "useana", "useina"],
    "Tra": ["muuksi", "muiksi", "moneksi", "moniksi", "useaksi", "useiksi"],
    "Ine": ["muussa", "muissa", "monessa", "monissa", "useassa", "useissa"],
    "Ela": ["muusta", "muista", "monesta", "monista", "useasta", "useista"],
    "Ill": ["muuhun", "muihin", "moneen", "moniin", "useaan", "useisiin"],
    "Ade": ["muulla", "muilla", "monelle", "monilla", "usealla", "useilla"],
    "Abl": ["muulta", "muilta", "monelta", "monilta", "usealta", "useilta"],
    "All": ["muille", "muille", "monelle", "monille", "usealle", "useille"]
}
ind3_numbers = ["Sing", "Plur", "Sing", "Plur", "Sing", "Plur"]
ind3_lemmas = ["muu", "muu", "moni", "moni", "usea", "usea"]


ind4_pron = {
    "Nom": ["molemmat", "itse"],
    "Gen": ["molempien", "itsen"],
    "Par": ["molempia", "itseä"],
    "Ess": ["molempina", "itsenä"],
    "Tra": ["molemmiksi", "itseksi"],
    "Ine": ["molemmissa""itsessä"],
    "Ela": ["molemmista", "itsestä"],
    "Ill": ["molempiin", "itseen"],
    "Ade": ["molemmilla", "itsellä"],
    "Abl": ["molemmilta", "itseltä"],
    "All": ["molemmille", "itselle"]
}
ind4_numbers = ["Plur", "Sing"]
ind4_lemmas = ["molemmat", "itse"]


pron_morphs = build_morphs("Prs", prs_pron, prs_lemmas, prs_numbers, prs_persons)
pron_morphs.update(build_morphs("Dem", dem_pron, dem_lemmas, dem_numbers))
pron_morphs.update(build_morphs("Rel", rel_pron, rel_lemmas, rel_numbers))
pron_morphs.update(build_morphs("Int", int_pron, int_lemmas, int_numbers))
pron_morphs.update(build_morphs("Ind", ind_pron, ind_lemmas, ind_numbers))
pron_morphs.update(build_morphs("Ind", ind2_pron, ind2_lemmas, ind2_numbers))
pron_morphs.update(build_morphs("Ind", ind3_pron, ind3_lemmas, ind3_numbers))
pron_morphs.update(build_morphs("Ind", ind4_pron, ind4_lemmas, ind4_numbers))
pron_morphs.update({
    # Just Nom, because all inflected forms are identical to the forms of "mikään"
    "mitkään": morph_rule("mitkään", "Ind", "Plur", "Nom")
})

MORPH_RULES = {
    "Pron": pron_morphs
}

for tag, rules in MORPH_RULES.items():
    for key, attrs in dict(rules).items():
        rules[key.title()] = attrs
