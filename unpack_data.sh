#!/usr/bin/env bash
# Reassemble and unpack the dataset after cloning.
#
# The pipeline inputs and scored outputs are stored as a split tar archive in
# data_archive/. They are split at 95 MB because GitHub rejects any single file
# over 100 MB; splitting keeps the repository free of Git LFS entirely.
#
# Parquet is already compressed, so the archive is stored uncompressed --
# gzipping it made it larger in testing, not smaller.
#
# Run once, from the repository root:
#     ./unpack_data.sh
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d data_archive ]; then
  echo "error: data_archive/ not found. Run this from the repository root." >&2
  exit 1
fi

parts=(data_archive/dataset.tar.part-*)
if [ ! -e "${parts[0]}" ]; then
  echo "error: no archive parts found in data_archive/." >&2
  echo "If you cloned with a partial checkout, fetch them and retry." >&2
  exit 1
fi

echo "reassembling ${#parts[@]} parts and unpacking..."
cat "${parts[@]}" | tar xf -

echo
echo "unpacked:"
for d in major_project_2/data major_project_2/outputs fii_risk_engine/outputs; do
  [ -d "$d" ] && printf "  %-34s %s\n" "$d" "$(du -sh "$d" | cut -f1)"
done
echo
echo "Done. The pipelines can now run in place."
echo "Note: quarantined artifacts (PREAUDIT, BASELINE, _smoke, _trunc, _unit)"
echo "are deliberately NOT in the archive -- they predate the audit that"
echo "rebuilt the state object and must never enter a reported number."
