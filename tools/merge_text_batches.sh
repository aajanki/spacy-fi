#!/bin/bash

set -euo pipefail

find "$1" -name '*.txt' | sort | xargs cat > "$2"
