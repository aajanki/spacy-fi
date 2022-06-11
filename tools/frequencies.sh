#!/bin/bash

set -euo pipefail

bzcat "$1"/* | python -m tools.frequencies - | LC_ALL=C sort -k 1nr -k 2g | gzip > "$2"
