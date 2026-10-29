#!/usr/bin/env bash
# Rebuild WRITEUP.pdf from WRITEUP.md: pandoc to HTML with docs/writeup.css, then headless Chrome to PDF.
# Usage: docs/build_pdf.sh [path to chrome]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
CHROME=${1:-google-chrome}
pandoc WRITEUP.md -f markdown-implicit_figures -t html5 -s --katex=https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/ \
  --css docs/writeup.css --metadata pagetitle="Scale separation" -o writeup_tmp.html
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=15000 --print-to-pdf="$ROOT/WRITEUP.pdf" "file://$ROOT/writeup_tmp.html"
rm -f writeup_tmp.html
