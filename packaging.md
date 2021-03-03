# Packaging

Remember to change the version in meta.json!

```sh
mkdir -p packages
spacy package models/taggerparser/model-best packages --code fi/fi.py --meta-path fi/meta.json --create-meta --build sdist,wheel --force
```

## Publishing

```sh
git tag v0.5.0
git push --tags

twine upload packages/fi_experimental_web_md-0.5.0/dist/
```
