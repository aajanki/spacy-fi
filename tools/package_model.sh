#!/bin/sh

set -eu

TRAINED_MODEL=$1

mkdir -p packages
rm -rf packages/*
spacy package "$TRAINED_MODEL" packages --code fi/fi.py --meta-path fi/meta.json --create-meta --build none --force

PACKAGE_DIR=$(ls -d packages/*/fi_*)
NEW_PACKAGE_DIR=$(echo "$PACKAGE_DIR" | sed -E 's#(.*?)/#/\1/spacy_#')
mv "$PACKAGE_DIR" "$NEW_PACKAGE_DIR"

echo "Building the package"
cp python_packaging/setup.py packages/*/

(
    cd packages/*/;
    python setup.py sdist bdist_wheel
)
