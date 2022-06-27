#!/bin/bash

set -euo pipefail

find "$1" -name '*.txt.bz2' | \
  sort | \
  xargs bzcat | \
  python -m tools.tokenize_fi_one - - | \
  python -m tools.frequencies - | \
  LC_ALL=C sort -k 1nr -k 2g | \
  gzip > "$2"
