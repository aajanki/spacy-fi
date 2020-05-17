# Packaging

Package just the POS tagger and dependency parser (this is the model published on GitHub).

Remember to change the version below!

```sh
tools/package_model.sh models/taggerparser/model-best <<EOF
fi
experimental_web_md
0.3.0

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
git tag v0.3.0
git push --tags
```

Create a new release at
[https://github.com/aajanki/spacy-fi/releases](https://github.com/aajanki/spacy-fi/releases). Upload
models/python-package/fi_experimental_web_md-\*/dist/\*.whl to the
release.

Update the pip install link on the [README](README.md).
