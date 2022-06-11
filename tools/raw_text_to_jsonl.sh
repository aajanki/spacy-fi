#!/bin/bash

set -euo pipefail

find "$1" -name '*.txt.bz2' | \
  sort | \
  xargs bzcat | \
  python -m tools.raw_text_to_jsonl --limit "$3" - "$2"
