# Packaging

Remember to change the version in [fi/meta.json](fi/meta.json)!

Update the Changelog.

```sh
tools/package_model.sh training/merged
```

Optionally, to override the default spaCy compatibility specification,
add a new spec as the second parameter:

```sh
tools/package_model.sh training/merged ">=3.0.0,<3.2.0"
```

## Publishing

```sh
git tag v0.5.0
git push --tags

twine upload --repository spacy-fi-experimental-web-md packages/fi_experimental_web_md-0.5.0/dist/*
```

Create a [new release](https://github.com/aajanki/spacy-fi/releases/new) at Github.
