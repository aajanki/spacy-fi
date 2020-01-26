# Experimental Finnish language model for SpaCy

Finnish language model for [SpaCy](https://spacy.io/). The model contains POS tagger, dependency parser, word vectors, token frequencies and a lemmatizer (libvoikko). See below for notes about NER.

## Install the Finnish language model

First, install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

Next, install the model by running:
```
pip install https://github.com/aajanki/spacy-fi/releases/download/v0.2.0/fi_experimental_web_md-0.2.0-py3-none-any.whl
```

## Usage

```python
import spacy

nlp = spacy.load('fi_experimental_web_md')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

## Updating the model

### Setup a development environment

Install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

tools/download_data.sh
```

### Train the model

```
tools/train.sh
```

### Tests

```
tools/tests.sh
```

### Build a Python package

Package just the POS tagger and dependency parser (this is the model published on GitHub):

```
tools/package_model.sh models/taggerparser/model-best
```

Alternatively, to build a model with combined tagger, parser and NER capabilities, run the following:

```
tools/package_model.sh models/merged
```

Notes about the NER model:
* The model is trained on a very specific domain (technology news) and its out-of-domain generalization is quite poor.
* Distributing the NER model might not be possible because the training data license (CC BY-ND-NC) is incompatible with the lemmatizer license (GPL).

## License

All the content in this repository is available under the [GNU General Public License, version 3 or any later version](LICENSE). 

Source code and other files under fi and tools directories are additionally available under the [MIT license](LICENSE.MIT).

### License for the trained models (Python packages)

The trained models in https://github.com/aajanki/spacy-fi/releases are distributed under GPL v3.

### License for the training data

The data sets downloaded by the tools/download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): CC BY-SA
* [finer-data](https://github.com/mpsilfve/finer-data): The Digitoday material is licensed under CC Attribution-NoDerivs-NonCommercial (CC BY-ND-NC 1.0) and the Wikipedia material is licensed under Attribution-ShareAlike (CC BY-SA 3.0)
