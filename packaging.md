# Packaging

Remember to change the version in [fi/meta.json](fi/meta.json)!

```sh
tools/package_model.sh models/taggerparser/model-best
```

## Publishing

```sh
git tag v0.5.0
git push --tags

twine upload packages/fi_experimental_web_md-0.5.0/dist/*
```
