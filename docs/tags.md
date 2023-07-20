# Labels

## Part-of-speech

These are the possible values assigned for `token.pos`:

| POS   | Explanation               |
|:------|:--------------------------|
| ADJ   | adjective                 |
| ADP   | adposition                |
| ADV   | adverb                    |
| AUX   | auxiliary verb            |
| CCONJ | coordinating conjunction  |
| INTJ  | interjection              |
| NOUN  | noun                      |
| NUM   | numeral                   |
| PRON  | pronoun                   |
| PROPN | proper noun               |
| PUNCT | punctuation               |
| SCONJ | subordinating conjunction |
| SPACE | space                     |
| SYM   | symbol                    |
| VERB  | verb                      |
| X     | other, e.g. foreing       |

## Dependency tags

These are the possible values for `token.dep`:

| dep          | Explanation                                                                                                                                 |
|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| acl          | [clausal modifier of noun](https://universaldependencies.org/fi/dep/acl.html)                                                               |
| acl:relcl    | [relative clause modifier](https://universaldependencies.org/fi/dep/acl-relcl.html)                                                         |
| advcl        | [adverbial clause modifier](https://universaldependencies.org/fi/dep/advcl.html)                                                            |
| advmod       | [adverb modifier](https://universaldependencies.org/fi/dep/advmod.html)                                                                     |
| amod         | [adjectival modifier](https://universaldependencies.org/fi/dep/amod.html)                                                                   |
| appos        | [apposition](https://universaldependencies.org/fi/dep/appos.html)                                                                           |
| aux          | auxiliary verb. One of the following: *olla*, *ei*, *voida*, *pitää*, *saattaa*, *täytyä*, *joutua*, *aikoa*, *taitaa*, *tarvita*, *mahtaa* |
| aux:pass     | [passive auxiliary](https://universaldependencies.org/fi/dep/aux-pass.html), only one possible verb: *olla*                                 |
| case         | [case marking](https://universaldependencies.org/fi/dep/case.html)                                                                          |
| cc           | [coordinating conjunction](https://universaldependencies.org/fi/dep/cc.html)                                                                |
| ccomp        | [clausal complement](https://universaldependencies.org/fi/dep/ccomp.html)                                                                   |
| cc:preconj   | [preconjunct](https://universaldependencies.org/fi/dep/cc-preconj.html), constructs like *sekä ... että*                                    |
| compound     | [compound](https://universaldependencies.org/fi/dep/compound.html)                                                                          |
| compound:nn  | [noun compound modifier](https://universaldependencies.org/fi/dep/compound-nn.html)                                                         |
| compound:prt | [phrasal particle](https://universaldependencies.org/fi/dep/compound-prt.html)                                                              |
| conj         | [coordinated element](https://universaldependencies.org/fi/dep/conj.html)                                                                   |
| cop          | [copula](https://universaldependencies.org/fi/dep/cop.html), *auto on vihreä*                                                               |
| cop:own      | [copula for posessive clauses](https://universaldependencies.org/fi/dep/cop-own.html), *minulla on kynä*                                    |
| csubj        | [clausal subject](https://universaldependencies.org/fi/dep/csubj.html)                                                                      |
| csubj:cop    | [clausal copular subject](https://universaldependencies.org/fi/dep/csubj-cop.html)                                                          |
| det          | [determiner](https://universaldependencies.org/fi/dep/det.html)                                                                             |
| dep          | [unspecified dependency](https://universaldependencies.org/u/dep/dep.html)                                                                  |
| discourse    | [discourse element](https://universaldependencies.org/fi/dep/discourse.html)                                                                |
| dislocated   | [dislocated elements](https://universaldependencies.org/u/dep/dislocated.html)                                                              |
| fixed        | [fixed multi-word expression](https://universaldependencies.org/fi/dep/fixed.html)                                                          |
| flat         | [flat phrase without a clear head](https://universaldependencies.org/fi/dep/flat.html)                                                      |
| flat:foreign | [foreign words](https://universaldependencies.org/u/dep/flat-foreign.html)                                                                  |
| flat:name    | [names](https://universaldependencies.org/u/dep/flat-name.html)                                                                             |
| goeswith     | [relation that links two parts of a compound word that are erroneously separated](https://universaldependencies.org/fi/dep/goeswith.html)   |
| mark         | [subordinating conjunction, complementizer, or comparative conjunction](https://universaldependencies.org/fi/dep/mark.html)                 |
| nmod         | [nominal modifier](https://universaldependencies.org/fi/dep/nmod.html)                                                                      |
| nmod:gobj    | [genitive object](https://universaldependencies.org/fi/dep/nmod-gobj.html)                                                                  |
| nmod:gsubj   | [genitive subject](https://universaldependencies.org/fi/dep/nmod-gsubj.html)                                                                |
| nmod:poss    | [genitive modifier](https://universaldependencies.org/fi/dep/nmod-poss.html)                                                                |
| nsubj        | [nominal subject](https://universaldependencies.org/fi/dep/nsubj.html)                                                                      |
| nsubj:cop    | [nominal copular subject](https://universaldependencies.org/fi/dep/nsubj-cop.html)                                                          |
| nsubj:outer  | [outer clause nominal subject](https://universaldependencies.org/u/dep/nsubj-outer.html)                                                    |
| nummod       | [numeric modifier](https://universaldependencies.org/fi/dep/nummod.html)                                                                    |
| obj          | [direct object](https://universaldependencies.org/fi/dep/obj.html)                                                                          |
| obl          | [oblique nominal](https://universaldependencies.org/u/dep/obl.html)                                                                         |
| orphan       | [orphaned dependent in gapping](https://universaldependencies.org/fi/dep/orphan.html)                                                       |
| parataxis    | [parataxis](https://universaldependencies.org/fi/dep/parataxis.html)                                                                        |
| punct        | [punctuation](https://universaldependencies.org/fi/dep/punct.html)                                                                          |
| root         | [grammatical root of the sentence](https://universaldependencies.org/fi/dep/root.html)                                                      |
| vocative     | [vocative modifier](https://universaldependencies.org/fi/dep/vocative.html)                                                                 |
| xcomp        | [open clausal complement](https://universaldependencies.org/fi/dep/xcomp.html)                                                              |
| xcomp:ds     | [clausal complement with different subject](https://universaldependencies.org/fi/dep/xcomp-ds.html)                                         |

## Morphology

The morphology labels (`token.morph`) follow the [UD for Finnish](https://universaldependencies.org/fi/index.html#morphology) specification.

## Named entities

The recognized named entities (`token.ent_type`) follow the OntoNotes scheme:

| ent\_type     | Explanation                                          |
|:--------------|:-----------------------------------------------------|
| CARDINAL      | Numerals that do not fall under another type         |
| DATE          | Absolute or relative dates or periods                |
| EVENT         | Named hurricanes, battles, wars, sports events, etc. |
| FAC           | Buildings, airports, highways, bridges, etc.         |
| GPE           | Geo-political entity: Countries, cities, states      |
| LANGUAGE      | Any named language                                   |
| LAW           | Named documents made into laws                       |
| LOC           | Non-GPE locations, mountain ranges, bodies of water  |
| MONEY         | Monetary values, including unit                      |
| NORP          | Nationalities or religious or political groups       |
| ORDINAL       | Ordinal numbers: *ensimmäinen*, *toinen*, etc.       |
| ORG           | Companies, agencies, institutions, etc.              |
| PERCENT       | Percentage (including “%”)                           |
| PERSON        | People, including fictional                          |
| PRODUCT       | Vehicles, weapons, foods, etc. (Not services)        |
| QUANTITY      | Measurements, as of weight or distance               |
| TIME          | Times smaller than a day                             |
| WORK\_OF\_ART | Titles of books, songs, etc.                         |

