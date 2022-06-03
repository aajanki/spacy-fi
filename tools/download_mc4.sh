#!/bin/bash

set -euo pipefail

python -m tools.download_huggingface Finnish-NLP/mc4_fi_cleaned Finnish-NLP--mc4_fi_cleaned "$1" - | \
  python -m tools.cleanup_mc4_punctuation | \
  bzip2 > "$2"
