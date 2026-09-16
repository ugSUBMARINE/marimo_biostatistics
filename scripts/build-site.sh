#!/usr/bin/env bash
# Run from any directory. An optional first argument selects the output directory.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_dir="${1:-_site}"

mkdir -p "$output_dir"
cp -R site/. "$output_dir/"

# marimo's html-wasm exporter shares "$output_dir/public/wheels" across every
# notebook below and deletes any locally-bundled module wheel
# (companion_*.py) not needed by the notebook it is currently exporting, so a
# wheel needed only by an earlier notebook can be pruned by a later one.
# Wheel filenames are content-hashed and therefore stable within a build, so
# collect every wheel produced during the loop in a scratch directory and
# restore the union into public/wheels once the loop finishes.
wheel_cache="$(mktemp -d)"
trap 'rm -rf "$wheel_cache"' EXIT
shopt -s nullglob

for notebook in 0[1-6]_*.py; do
  uv run --locked marimo export html-wasm "$notebook" \
    --output "$output_dir/${notebook%.py}.html" \
    --mode run \
    --no-sandbox \
    --force

  wheels=("$output_dir"/public/wheels/*.whl)
  if [ "${#wheels[@]}" -gt 0 ]; then
    # Plain cp (not -n): wheel filenames are content-hashed, so a name
    # collision means identical content, and overwriting is a safe no-op.
    # `cp -n`'s exit status on an existing destination differs across
    # platforms (e.g. macOS cp returns 1), which would abort under -e.
    cp "${wheels[@]}" "$wheel_cache"/
  fi
done

mkdir -p "$output_dir/public/wheels"
cached_wheels=("$wheel_cache"/*.whl)
if [ "${#cached_wheels[@]}" -gt 0 ]; then
  cp "${cached_wheels[@]}" "$output_dir/public/wheels/"
fi

printf 'Website built in %s\n' "$output_dir"
