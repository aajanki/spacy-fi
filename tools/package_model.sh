#!/bin/sh

set -eu

TRAINED_MODEL=$1

rm -rf models/python-package/*
mkdir -p models/python-package
spacy package "$TRAINED_MODEL" models/python-package --create-meta --force

PACKAGE_DIR=`ls models/python-package`
ORIG_MODEL_NAME=`echo $PACKAGE_DIR | sed 's/-[0-9.]\+$//'`
MODEL_NAME="spacy_$ORIG_MODEL_NAME"

mv "models/python-package/$PACKAGE_DIR/$ORIG_MODEL_NAME" "models/python-package/$PACKAGE_DIR/$MODEL_NAME"

echo "Copying the lemmatizer sources to the package directory"
mkdir -p models/python-package/"$PACKAGE_DIR/$MODEL_NAME"
cp -r fi/[^_]*.py fi/lookups/ models/python-package/"$PACKAGE_DIR/$MODEL_NAME"/

echo "Adding import to __init__.py"
cat python_packaging/init_extra.py >> models/python-package/"$PACKAGE_DIR/$MODEL_NAME"/__init__.py

echo "Updating requirements in meta.json"
jq '.requirements = ["voikko>=0.5"]' \
   < models/python-package/"$PACKAGE_DIR"/meta.json \
   > /tmp/fi_exp_meta.json
mv /tmp/fi_exp_meta.json models/python-package/"$PACKAGE_DIR"/meta.json


echo "Building the package"
cp python_packaging/setup.py models/python-package/"$PACKAGE_DIR"

(
    cd models/python-package/"$PACKAGE_DIR";
    python setup.py sdist bdist_wheel
)
