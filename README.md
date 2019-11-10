Experimental Finnish language model for [SpaCy](https://spacy.io/).

## Setup

Install libvoikko with Finnish morphology data files.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./download_data.sh
```

## License

The source code and all other material in this repository is licensed
under the MIT license.

The data sets downloaded by download_data.sh script are licensed as follows:
* [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
* Word vectors and frequencies at [Finnish Internet Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank): ???
