#!/usr/bin/env bash
# Run from any directory. An optional first argument selects the output directory.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_dir="${1:-_site}"

mkdir -p "$output_dir"
cp -R site/. "$output_dir/"

for notebook in 0[1-6]_*.py; do
  uv run --locked marimo export html-wasm "$notebook" \
    --output "$output_dir/${notebook%.py}.html" \
    --mode run \
    --no-sandbox \
    --force
done

printf 'Website built in %s\n' "$output_dir"
