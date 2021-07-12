# Experimental Finnish language model for spaCy

Finnish language model for [spaCy](https://spacy.io/). The model does POS tagging, dependency parsing, word vectors, noun phrase extraction, token frequencies, morphological features and lemmatization. The morphological features and lemmatization are based on [Voikko](https://voikko.puimula.org/).

## Install the Finnish language model

First, install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

Next, install the model by running:
```
pip install spacy_fi_experimental_web_md
```

Compatibility with spaCy versions:

| spacy-fi version | Compatible with spaCy version |
| ---------------- | ----------------------------- |
| 0.7.0            | 3.0.x, 3.1.x                  |
| 0.6.0            | 3.0.x                         |
| 0.5.0            | 3.0.x                         |
| 0.4.x            | 2.3.x                         |

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

```sh
# Install the libvoikko native library with Finnish morphology data.
#
# This will install Voikko on Debian/Ubuntu.
# For other distros and operating systems, see https://voikko.puimula.org/python.html
sudo apt install libvoikko1 voikko-fi

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

To perform full pretraining (slow!), append the `--pretrain` option:

```sh
tools/train.sh --pretrain
```

### Tests

```
python -m pytest tests
```

Loading the trained model locally without packaging it as a module:

```python
import spacy
import fi

nlp = spacy.load('models/taggerparser/model-best/')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

### What about named entity recognizer (NER)?

An earlier version of this script optinally trained a NER model, but
the current version does not. Mostly because the model was never very
good and the training data potentially has licensing issues.

### Packaging and publishing

See [packaging.md](packaging.md).

## License

[MIT license](LICENSE)

### License for the training data

The data sets downloaded by the tools/download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): CC BY-SA
* [finer-data](https://github.com/mpsilfve/finer-data): The Digitoday material is licensed under CC Attribution-NoDerivs-NonCommercial (CC BY-ND-NC 1.0) and the Wikipedia material is licensed under Attribution-ShareAlike (CC BY-SA 3.0)
