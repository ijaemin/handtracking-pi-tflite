#!/usr/bin/env bash
# Usage: ./make_ppm.sh input.jpg output.ppm
set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 input.jpg output.ppm"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

python3 - <<PYCODE
from PIL import Image

# Load image from shell-provided paths
im = Image.open("$INPUT").convert("RGB").resize((300, 300))
im.save("$OUTPUT", format="PPM")
print("Saved PPM:", "$OUTPUT")
PYCODE