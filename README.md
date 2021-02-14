# Experimental Finnish language model for spaCy

Finnish language model for [spaCy](https://spacy.io/). The model contains POS tagger, dependency parser, word vectors, noun phrase extraction, token frequencies and a lemmatizer (libvoikko). See below for notes about NER.

## Install the Finnish language model

First, install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

Next, install the model by running:
```
pip install spacy_fi_experimental_web_md
```

## Usage

```python
import spacy

nlp = spacy.load('spacy_fi_experimental_web_md')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

## Updating the model

### Setup a development environment

Install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install wheel
pip install -r requirements.txt

tools/download_data.sh
```

### Train the model

```sh
tools/train.sh
```

### Tests

```
python -m pytest tests
```

Loading the trained model locally without packaging it as a module:

```python
import spacy
from fi.fi import FinnishEx

spacy.util.set_lang_class('fi', FinnishEx)

nlp = spacy.load('models/merged')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

### Notes about the NER model

* The model is trained on a very specific domain (technology news) and its out-of-domain generalization is quite poor.
* Distributing the NER model might not be possible because the training data license (CC BY-ND-NC) is incompatible with the lemmatizer license (GPL).

### Packaging and publishing

See [packaging.md](packaging.md).

## License

[MIT license](LICENSE)

### License for the training data

The data sets downloaded by the tools/download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): CC BY-SA
* [finer-data](https://github.com/mpsilfve/finer-data): The Digitoday material is licensed under CC Attribution-NoDerivs-NonCommercial (CC BY-ND-NC 1.0) and the Wikipedia material is licensed under Attribution-ShareAlike (CC BY-SA 3.0)
