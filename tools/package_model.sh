#!/bin/sh

set -eu

TRAINED_MODEL=$1

rm -rf models/python-package/*
mkdir -p models/python-package
spacy package "$TRAINED_MODEL" models/python-package --create-meta --force

PACKAGE_DIR=`ls models/python-package`
MODEL_NAME=`echo $PACKAGE_DIR | sed 's/-[0-9.]\+$//'`


echo "Copying the lemmatizer sources to the package directory"
mkdir -p models/python-package/"$PACKAGE_DIR/$MODEL_NAME"
cp fi/fi.py fi/lemmatizer.py fi/punctuation.py models/python-package/"$PACKAGE_DIR/$MODEL_NAME"/
cp -r fi/lookups/ models/python-package/"$PACKAGE_DIR/$MODEL_NAME"/

echo "Adding import to __init__.py"
cat - >> models/python-package/"$PACKAGE_DIR/$MODEL_NAME"/__init__.py <<EOF



from spacy.util import set_lang_class
from .fi import FinnishEx, FinnishExDefaults
from .lemmatizer import FinnishLemmatizer

set_lang_class('fi', FinnishEx)
EOF


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
