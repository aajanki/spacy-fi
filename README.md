[![CI status](https://circleci.com/gh/aajanki/spacy-fi/tree/master.svg?style=shield)](https://circleci.com/gh/aajanki/spacy-fi/tree/master)

# Experimental Finnish language model for spaCy

Finnish language model for [spaCy](https://spacy.io/). The model does POS tagging, dependency parsing, word vectors, noun phrase extraction, word occurrence probability estimates, morphological features, lemmatization and named entity recognition (NER). The lemmatization is based on [Voikko](https://voikko.puimula.org/).

The main differences between this model and the [Finnish language model](https://spacy.io/models/fi) in the spaCy core:
* This model includes a different lemmatizer implementation compared to spaCy core. My model's [lemmatization accuracy](https://github.com/aajanki/finnish-pos-accuracy#results) is considerably better but the execution speed is slightly lower.
* This model requires libvoikko. The spaCy model core does not need any external dependencies.
* The training data for this model is partly different, and there are other minor tweaks in the pipeline implementation.

Want a hassle free installation? Install the [spaCy core model](https://spacy.io/models/fi).
Need the highest possible accuracy especially for lemmatization? Install this model.

I'm planning to continue to experiment with new ideas on this repository and push the useful features to the spaCy core after testing them here.

The training data is web pages collected during 2014 - 2020, before the rise of the AI slop surge. The data does contain some regular spam and poorly machine-translated pages. I have made some effort to filter out the most conspicuous spam pages.

## Install the Finnish language model

First, install [the libvoikko native library and the Finnish morphology data files](https://voikko.puimula.org/python.html).

Next, install the model by running:
```
pip install spacy_fi_experimental_web_md
```

Compatibility with spaCy versions:

| spacy-fi version | Compatible with spaCy versions |
|------------------|--------------------------------|
| 0.15.x           | 3.8.x                          |
| 0.14.0           | 3.7.x                          |
| 0.13.0           | 3.6.x                          |
| 0.12.0           | 3.5.x                          |
| 0.11.0           | 3.4.x                          |
| 0.10.0           | 3.3.x                          |
| 0.9.0            | >= 3.2.1 and < 3.3.0           |
| 0.8.x            | 3.2.x                          |
| 0.7.x            | 3.0.x, 3.1.x                   |
| 0.6.0            | 3.0.x                          |
| 0.5.0            | 3.0.x                          |
| 0.4.x            | 2.3.x                          |

## Usage

```python
import spacy

nlp = spacy.load('spacy_fi_experimental_web_md')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

The [dependency, part-of-speech and named entity labels](docs/tags.md) are documented on a separate page.

## Updating the model

### Setting up a development environment

```sh
# Install the libvoikko native library with Finnish morphology data.
#
# This will install Voikko on Debian/Ubuntu.
# For other distros and operating systems, see https://voikko.puimula.org/python.html
sudo apt install libvoikko1 voikko-fi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training the model

```sh
spacy project assets
spacy project run train-pipeline
```

Optional steps (slow!) for training certain model components. These
steps are not necessarily required because the results of have been
pre-computed and stored in git.

Train floret embeddings:
```sh
spacy project run floret-vectors
```

Pretrain tok2vec weights:
```sh
spacy project run pretrain
```

Plot the pretraining loss:
```sh
python tools/plot_pretrain_loss.py training/pretrain/log.jsonl
```

### Testing

Unit tests:
```
python -m pytest tests/unit
```

Functional tests for a trained model:
```
python -m pytest tests/functional
```

Importing the trained model directly from the file system without
packaging it as a module:

```python
import spacy
import fi

nlp = spacy.load('training/merged')

doc = nlp('Hän ajoi punaisella autolla.')
for t in doc:
    print(f'{t.lemma_}\t{t.pos_}')
```

### Packaging and publishing

See [packaging.md](docs/packaging.md).

## License

[MIT license](LICENSE)

### Licenses for the training data

The datasets used in training are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* [TurkuONE](https://github.com/TurkuNLP/turku-one): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* [MC4](https://huggingface.co/datasets/allenai/c4): [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) and [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/)
