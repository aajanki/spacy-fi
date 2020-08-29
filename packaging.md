# Packaging

Package just the POS tagger and dependency parser (this is the model published on GitHub).

Remember to change the version below!

```sh
tools/package_model.sh models/taggerparser/model-best <<EOF
fi
experimental_web_md
0.4.1

Finnish language model: POS tagger, dependency parser, lemmatizer
Antti Ajanki
antti.ajanki@iki.fi
https://github.com/aajanki/spacy-fi
GPL v3.0
EOF
```

Alternatively, to build a model with combined tagger, parser and NER capabilities, run the following:

```sh
tools/package_model.sh models/merged
```

## Publishing

```sh
git tag v0.4.1
git push --tags

twine upload models/python-package/fi_experimental_web_md-0.4.1/dist/*
```
