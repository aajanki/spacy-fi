#!/bin/bash

set -euo pipefail

bzcat "$1"/* | python -m tools.frequencies - | sort -n | gzip > "$2"
