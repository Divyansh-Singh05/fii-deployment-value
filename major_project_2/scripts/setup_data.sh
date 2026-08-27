#!/usr/bin/env bash
# Extract the research data exports into data/.
# Source zips: Google Drive export of the two research folders.
# Override SOURCE_DIR if the zips live elsewhere.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$HOME/Desktop/Major Project 1/Data}"

VZIP="$SOURCE_DIR/FII_Module1_findings.zip"          # contains VALIDATION_DATA/
IZIP="$SOURCE_DIR/ISIN_MAPPING-20260711T193601Z-2-001.zip"

for z in "$VZIP" "$IZIP"; do
  [ -f "$z" ] || { echo "missing: $z"; exit 1; }
done

mkdir -p "$REPO/data"
echo "extracting VALIDATION_DATA ..."
unzip -qo "$VZIP" -d "$REPO/data"
echo "extracting ISIN_MAPPING ..."
unzip -qo "$IZIP" -d "$REPO/data"

echo "done:"
du -sh "$REPO/data/VALIDATION_DATA" "$REPO/data/ISIN_MAPPING"
