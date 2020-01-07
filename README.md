# Experimental Finnish language model for SpaCy

## Setup

Install [the libvoikko native library with Finnish morphology data files](https://voikko.puimula.org/python.html).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

tools/download_data.sh
```

## Train the model

```
tools/prepare_lemmas.sh
tools/train.sh
tools/package_model.sh models/taggerparser/model-best  # package only the POS tagger
```

Alternatively, to build a model with combined POS tagger and NER capabilities, run the following. Note that this package can't be distributed because of incompatible source data licenses.

```
tools/package_model.sh models/merged
```

## Usage

```
import spacy
import spacy.util
from fi_experimental_web_md import FinnishEx

spacy.util.set_lang_class('fi', FinnishEx)
nlp = spacy.load('fi_experimental_web_md')
```

## License

All the content in this repository is available under the [GNU General Public License, version 3 or any later version](LICENSE). The generated Python package (which includes libvoikko) is also licensed as GPL v3.

Source code and other files under fi, lemma and tools directories is additionally available under the [MIT license](LICENSE.MIT).

The data sets downloaded by download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): CC BY-SA
* [finer-data](https://github.com/mpsilfve/finer-data): The Digitoday material is licensed under CC Attribution-NoDerivs-NonCommercial (CC BY-ND-NC 1.0) and the Wikipedia material is licensed under Attribution-ShareAlike (CC BY-SA 3.0)
