# Experimental Finnish language model for SpaCy

Finnish language model for [SpaCy](https://spacy.io/). The model contains POS tagger, dependency parser, word vectors, token frequencies, lemmatizer (libvoikko). See below for notes about NER.

## Install the Finnish language model

Install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

```
pip install models/python-package/fi_experimental_web_md-0.0.1/dist/fi_experimental_web_md-0.0.1-py3-none-any.whl
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

### Development environment setup

Install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

tools/download_data.sh
```

### Training the model

```
tools/train.sh
```

### Build a Python package

Package just the POS tagger and dependency parser (this is the model published on PyPI):

```
tools/package_model.sh models/taggerparser/model-best
```

Alternatively, to build a model with combined tagger, parser and NER capabilities, run the following. Note that this package can't be distributed because of incompatible source data licenses.

```
tools/package_model.sh models/merged
```

## License

All the content in this repository is available under the [GNU General Public License, version 3 or any later version](LICENSE). The generated Python package (which includes libvoikko) is also licensed as GPL v3.

Source code and other files under fi and tools directories are additionally available under the [MIT license](LICENSE.MIT).

The data sets downloaded by download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): CC BY-SA
* [finer-data](https://github.com/mpsilfve/finer-data): The Digitoday material is licensed under CC Attribution-NoDerivs-NonCommercial (CC BY-ND-NC 1.0) and the Wikipedia material is licensed under Attribution-ShareAlike (CC BY-SA 3.0)
