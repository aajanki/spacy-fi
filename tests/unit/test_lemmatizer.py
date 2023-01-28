import pytest
from fi.fi import create_lookups_from_json_reader, VoikkoLemmatizer
from pathlib import Path
from spacy.lang.fi import Finnish
from spacy.tokens import Doc


XFAIL = 1
testcases = {
    'NOUN': [
        ('tila', ['tila']),
        ('tilassa', ['tila']),
        ('tilakaan', ['tila']),
        ('tilassakaan', ['tila']),
        ('tilassammekaan', ['tila']),
        ('tilasi', ['tila']),
        ('tilamme', ['tila']),
        ('rakkaasi', ['rakas', 'rakka']),
        ('opettajiimme', ['opettaja']),
        ('marjaansa', ['marja']),
        ('marjaksemme', ['marja']),
        ('isäksesi', ['isä']),
        ('sahaansa', ['saha']),
        ('sahojensa', ['saha']),
        ('taloihinsa', ['talo']),
        ('hahmoaan', ['hahmo']),
        ('kukissaan', ['kukka']),
        ('öissä', ['yö']),
        ('ylitöissä', ['ylityö']),
        ('emäntien', ['emäntä', 'emäntie']),
        ('perusteluissa', ['perustelu', 'perusteluu']),
        ('esittelijä', ['esittelijä']),
        ('esittelijät', ['esittelijä']),
        ('tilanne', ['tilanne', 'tila']),
        ('ensi-ilta', ['ensi-ilta']),
        ('elokuvan', ['elokuva']),
        ('tietoja', ['tieto']),
        ('aika', ['aika']),
        ('tuli', ['tuli']),
        ('löytäminen', ['löytäminen']),
        ('teitä', ['tie', 'tee']),
        ('epäjärjestelmällisyydellä', ['epäjärjestelmällisyys']),
        ('epäjärjestelmällisyydelläänkäänköhän', ['epäjärjestelmällisyys']),
        ('koko', ['koko']),
        ('kokkoko', ['kokko']),
        ('yksikkö', ['yksikkö']),
        ('yksikkökö', ['yksikkö']),
        ('leipä', ['leipä']),
        ('vanhemmaksi', ['vanhempi']),
        ('vanhempana', ['vanhempi']),
        ('perheenne', ['perhe']),
        ('ruokaa', ['ruoka']),
        ('tulevaisuudessa', ['tulevaisuus']),
        ('ihminen', ['ihminen']),

        # -minen
        ('ajaminenkaan', ['ajaminen']),
        ('testaamisessa', ['testaaminen']),
        ('yksipuolistuminen', ['yksipuolistuminen']),
        ('sulautumiseen', ['sulautuminen']),
        ('löytäminen', ['löytäminen']),
        ('opiskeleminen', ['opiskeleminen']),
        ('kuulemisiin', ['kuuleminen']),
        ('lukemisella', ['lukeminen']),
        ('häiritsemisemme', ['häiritseminen']),
        ('hyppäämiselläkään', ['hyppääminen']),
        ('välivaiheistuminen', ['välivaiheistuminen']),
        ('kaupallistuminenko', ['kaupallistuminen']),

        # compound words
        ('1500-luvulla', ['1500-luku']),
        ('ääri-ilmiöissä', ['ääri-ilmiö', 'ääri-ilmiyö']),
        ('ampumahiihtäjä', ['ampumahiihtäjä']),
        ('ampumahiihtäjäksi', ['ampumahiihtäjä']),
        ('urheilu-ura', ['urheilu-ura']),
        ('puku', ['puku']),
        ('tonttupukuihin', ['tonttupuku']),
        ('kuopistaan', ['kuoppa']),
        ('maakuopistaan', ['maakuoppa']),
        ('johdolle', ['johto']),
        ('tietohallintajohdolle', ['tietohallintajohto']),
        ('työlupa', ['työlupa']),
        ('työlupakaan', ['työlupa']),
        ('markkinavalvonnalla', ['markkinavalvonta']),
        ('yökerhossa', ['yökerho']),
        ('lähtökohdiltaan', ['lähtökohta']),
        ('voimakeinoja', ['voimakeino'], XFAIL),
        ('keskiluokkaa', ['keskiluokka']),
        ('kirjoitusasulla', ['kirjoitusasu']),
        ('kansalaisuuskäsitteenä', ['kansalaisuuskäsite']),
        ('laivapojilla', ['laivapoika']),
        ('valtameriin', ['valtameri']),
        ('VGA-kaapelia', ['VGA-kaapeli']),

        # gradation test cases
        # av1
        ('valikon', ['valikko']),
        ('maton', ['matto']),
        ('kaapin', ['kaappi']),
        ('ruukun', ['ruukku']),
        ('somman', ['sompa']),
        ('tavan', ['tapa']),
        ('kunnan', ['kunta']),
        ('killan', ['kilta']),
        ('kerran', ['kerta']),
        ('pöydän', ['pöytä']),
        ('hangon', ['hanko']),
        ('puvun', ['puku']),
        ('kyvyn', ['kyky']),
        # av2
        ('riitteen', ['riite']),
        ('oppaan', ['opas']),
        ('liikkeen', ['liike']),
        ('lumpeen', ['lumme']),
        ('tarpeen', ['tarve']),
        ('ranteen', ['ranne']),
        ('siveltimen', ['sivellin']),
        ('vartaan', ['varras']),
        ('sateen', ['sade']),
        ('kankaan', ['kangas']),
        # av3
        ('järjen', ['järki']),
        # av4
        ('palkeen', ['palje']),
        # av5
        ('vuoan', ['vuoka']),
        # av6
        ('säikeen', ['säie']),

        # abbreviations
        ('EU:ssa', ['EU']),
        ('STT:n', ['STT']),
        ('ry:ksi', ['ry']),
        ('YK:lle', ['YK']),
        ('4g:lle', ['4g']),
    ],

    'VERB': [
        # negation
        ('ei', ['ei']),
        ('en', ['ei']),
        ('emme', ['ei']),
        ('ettekö', ['ei']),
        ('älkää', ['ei']),

        # olla
        ('oli', ['olla']),
        ('olitte', ['olla']),
        ('ole', ['olla']),
        ('olisi', ['olla']),

        # person
        ('annan', ['antaa']),
        ('hukkaan', ['hukata']),
        ('lasket', ['laskea']),
        ('laskee', ['laskea']),
        ('punoo', ['punoa']),
        ('ampuu', ['ampua']),
        ('raahaa', ['raahata']),
        ('tulee', ['tulla']),
        ('häviätte', ['hävitä']),
        ('kitisevät', ['kitistä']),
        ('täytyy', ['täytyä']),
        ('grillailen', ['grillailla']),
        ('pyöritteli', ['pyöritellä']),

        # past tense
        ('kelluit', ['kellua']),
        ('tuli', ['tulla']),
        ('pinositte', ['pinota']),
        ('valaistuivat', ['valaistua']),
        ('jäi', ['jäädä']),
        ('möksähti', ['möksähtää'], XFAIL),
        ('pelastuimme', ['pelastua']),

        # past perfect
        ('kuullut', ['kuulla']),
        ('ansainneet', ['ansaita']),
        ('ennaltaehkäissyt', ['ennaltaehkäistä']),
        ('ilmakuivattu', ['ilmakuivata']),
        ('esiopettanut', ['esiopettaa']),
        ('rauhanturvanneet', ['rauhanturvata']),

        # conditional
        ('hakisi', ['hakea']),
        ('imartelisitte', ['imarrella']),
        ('karmisi', ['karmia']),

        # passive
        ('esitellään', ['esitellä']),
        ('opittiin', ['oppia']),
        ('supistaan', ['supista']),
        ('juodaan', ['juoda']),
        ('juhlittu', ['juhlia']),
        ('kadehdittukaan', ['kadehtia']),
        ('kiirehditty', ['kiirehtiä']),

        # imperative
        ('järisyttäköön', ['järisyttää']),
        ('astukoon', ['astua']),
        ('kadotkaamme', ['kadota']),
        ('polje', ['polkea']),
        ('valitse', ['valita']),
        ('epäröi', ['epäröidä']),
        ('hypätkää', ['hypätä']),
        ('kirjoittakoot', ['kirjoittaa']),
        ('kutoko', ['kutoa']),
        ('valehdelko', ['valehdella']),

        # A-infinitive
        ('pelastua', ['pelastua']),
        ('nähdäkseen', ['nähdä']),

        # NUT-participle
        ('kimpaantunut', ['kimpaantua']),
        ('pyytänyt', ['pyytää']),
        ('leikkinytkin', ['leikkiä']),
        ('kadehdittuja', ['kadehtia']),
        ('neuvotelleet', ['neuvotella']),

        # VA-participle
        ('valittava', ['valittaa'], XFAIL),
        ('lapioiva', ['lapioida']),
        ('kestävä', ['kestää']),
        ('häiritsevät', ['häiritä']),

        # agent participle
        ('ottama', ['ottaa']),
        ('keräämä', ['kerätä']),
        ('harrastama', ['harrastaa']),

        # enclitics
        ('pohdinko', ['pohtia']),
        ('lähdettehän', ['lähteä']),

        # gradation
        #av1
        ('ilkutte', ['ilkkua']),
        # av2
        ('lobbaatte', ['lobata']),
        ('diggaavat', ['digata'], XFAIL),
        # av3
        ('hyljit', ['hylkiä']),
        # av4
        ('ilkeät', ['iljetä']),
        # av5
        ('aion', ['aikoa']),
        # av6
        ('aukeat', ['aueta']),
    ],

    'ADJ': [
        ('lämmin', ['lämmin']),
        ('mielenkiintoinen', ['mielenkiintoinen']),
        ('normaaliin', ['normaali']),
        ('maalaamatontakin', ['maalaamaton']),
        ('kultaiset', ['kultainen']),
        ('punaisten', ['punainen']),
        ('sujuvilla', ['sujuva']),
        ('ranskalaisemme', ['ranskalainen']),
        ('ihanan', ['ihana']),
        ('onnistunut', ['onnistunut']),
        ('rajoittunutkin', ['rajoittunut']),
        ('tuohtunutta', ['tuohtunut']),
        ('valittava', ['valittava']),
        ('ilmielävääkään', ['ilmielävä']),
        ('ilmakuivattu', ['ilmakuivattu']),
        ('esiopetetut', ['esiopetettu']),

        # komparatiivi
        ('lämpimämpi', ['lämmin']),
        ('surullisempi', ['surullinen']),
        ('voimakkaampi', ['voimakas']),
        ('uudempi', ['uusi']),
        ('parempi', ['hyvä']),

        # superlatiivi
        ('lämpimin', ['lämmin']),
        ('kaunein', ['kaunis']),
        ('nopein', ['nopea']),
        ('lyhyin', ['lyhyt']),
        ('paras', ['hyvä']),
    ],

    'PROPN': [
        ('Etelä-Afrikassa', ['Etelä-Afrikka']),
        ('Hangosta', ['Hanko']),
        ('Belgiakin', ['Belgia']),
        ('Tampereeltamme', ['Tampere']),
        ('Annan', ['Anna']),
        ('Vihreät', ['Vihreä']),
    ],

    'ADV': [
        ('myös', ['myös']),
        ('vain', ['vain']),
        ('vuoksi', ['vuoksi']),
        ('tänäänkin', ['tänään']),
        ('kohta', ['kohta']),
        ('piankin', ['pian']),
        ('nopeasti', ['nopeasti']),
        ('nopeammin', ['nopeasti'], XFAIL),
        ('useasti', ['useasti']),
        ('fyysisestikin', ['fyysisesti']),
        ('luonnollisestikaan', ['luonnollisesti']),
        ('viidesti', ['viidesti']),
        ('tuhannestikin', ['tuhannesti']),
        ('tarpeeksi', ['tarpeeksi']),
        ('onneksikaan', ['onneksi']),
        ('onneksemme', ['onneksi']),
        ('aluksi', ['aluksi']),
        ('toiseksi', ['toiseksi']),
        ('kunnolla', ['kunnolla']),
        ('lopulta', ['lopulta']),
        ('täynnä', ['täynnä']),
        ('päällä', ['päällä']),
        ('päälle', ['päälle']),
        ('välillä', ['välillä']),
        ('varsinkin', ['varsinkin']),
        ('tosin', ['tosin']),
        ('tosiaan', ['tosiaan']),
        ('edelleen', ['edelleen']),
        ('edelleenkö', ['edelleen']),
        ('kanssamme', ['kanssa']),
        ('postitse', ['postitse']),
        ('järeämmin', ['järeämmin']),
        ('voimakkaamminkaanko', ['voimakkaammin']),
        ('voimakkaamminkokaan', ['voimakkaammin']),
        ('mukaani', ['mukaan']),
    ],

    'ADP': [
        ('alkaen', ['alkaen']),
        ('takaa', ['takaa']),
        ('sijaan', ['sijaan']),
        ('jälkeenkin', ['jälkeen']),
    ],

    'NUM': [
        ('nollakin', ['nolla']),
        ('neljäs', ['neljäs']),
        ('viiden', ['viisi']),
        ('viidenkin', ['viisi']),
        ('kymmentä', ['kymmenen']),
        ('kymmeniä', ['kymmenen']),
        ('kymmenes', ['kymmenes']),
        ('tuhannen', ['tuhat']),
        ('miljoonaa', ['miljoona']),
        ('kahdesta', ['kaksi']),
        ('yhtenä', ['yksi']),
        ('yhdelle', ['yksi']),
        ('kahdelle', ['kaksi']),
        ('kolmelle', ['kolme']),
        ('neljälle', ['neljä']),
        ('viidelle', ['viisi']),
        ('kuudelle', ['kuusi']),
        ('seitsemälle', ['seitsemän']),
        ('kahdeksalle', ['kahdeksan']),
        ('yhdeksälle', ['yhdeksän']),
        ('kymmenelle', ['kymmenen']),
        ('sadalle', ['sata']),
        ('tuhannelle', ['tuhat']),
        ('miljoonalle', ['miljoona']),
        ('4:s', ['4']),
        ('6:teen', ['6']),
        ('III', ['III']),
    ],

    'PRON': [
        ('sen', ['se']),
        ('siitä', ['se']),
        ('niiden', ['se']),
        ('hänet', ['hän']),
        ('jonkin', ['jokin']),
        ('teitä', ['sinä']),
        ('hänenkin', ['hän']),
        ('teitäkään', ['sinä']),
        ('minun', ['minä']),
        ('meitä', ['minä']),
        ('häneltä', ['hän']),
        ('häneltäkään', ['hän']),
        ('kehen', ['kuka']),
        ('kenenkään', ['kukaan']),
        ('multa', ['minä']),
        ('sut', ['sinä']),
        ('sulle', ['sinä']),
        ('ainoa', ['ainoa']),
        ('ainoamme', ['ainoa']),
        ('moniksi', ['moni']),
    ],

    'SCONJ': [
        ('sillä', ['sillä']),
    ],

    'SYM': [
        (':)', [':)']),
        (':D', [':D']),
        ('XXX', ['XXX']),
    ]
}


def check(cases, case_filter=None, accept_less_common=True):
    nlp = Finnish()
    lemmatizer = VoikkoLemmatizer(nlp.vocab)
    lemmatizer.initialize(lookups = create_lookups_from_json_reader(
        Path(__file__).parent.parent.parent / 'fi' / 'lookups' / 'lemmatizer'))

    expanded = []
    for pos, tokens in cases.items():
        for test_case in tokens:
            if (
                    (case_filter is None and len(test_case) == 2)
                    or
                    (case_filter is not None and len(test_case) == 3 and test_case[2] == case_filter)
            ):
                word, lemmas = test_case[:2]
                expanded.append((word, lemmas, pos))

    if len(expanded) == 0:
        return 0, 0.0

    failed = []
    for (word, lemmas, univ_pos) in expanded:
        if not accept_less_common:
            lemmas = lemmas[:1]

        doc = Doc(vocab=nlp.vocab, words=[word], pos=[univ_pos], deps=["ROOT"])
        observed = lemmatizer(doc)[0].lemma_
        if observed not in lemmas:
            failed.append((word, univ_pos, observed, lemmas))

    failed_proportion = len(failed)/len(expanded)
    return failed, failed_proportion


def test_lemmas():
    failed, failed_prop = check(testcases)

    assert len(failed) == 0


@pytest.mark.xfail
def test_lemmas_xfail():
    failed, failed_prop = check(testcases, case_filter=XFAIL)

    assert len(failed) == 0
