#!/bin/bash

set -euo pipefail

find "$1" -name '*.txt.bz2' | sort | xargs bzcat > "$2"
