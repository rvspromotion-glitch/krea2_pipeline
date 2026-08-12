#!/usr/bin/env bash
# Fetch the custom node packages listed in custom_nodes.txt and install their
# Python requirements into the active venv.
#
# Tarballs, not `git clone`: codeload gives a single compressed stream with no
# .git directory, which is both faster and smaller in the image. Nothing here
# ever updates in place — a rebuild refetches.
set -euo pipefail

LIST="${1:-/build/custom_nodes.txt}"
CUSTOM_NODES="${2:?custom_nodes dir required}"
CONSTRAINTS="${PIP_CONSTRAINT:-/build/constraints.txt}"

mkdir -p "$CUSTOM_NODES"

while read -r name repo ref; do
  case "$name" in ""|\#*) continue ;; esac
  dest="${CUSTOM_NODES}/${name}"
  echo "[nodes] ${name} <- ${repo}@${ref}"
  tmp="$(mktemp --suffix=.tar.gz)"
  curl -fSL --retry 5 --retry-delay 3 -o "$tmp" \
    "https://codeload.github.com/${repo}/tar.gz/${ref}"
  mkdir -p "$dest"
  tar -xzf "$tmp" -C "$dest" --strip-components=1
  rm -f "$tmp"
done < "$LIST"

# BatchnodeI9 and savezipi9 are first-party and ship their node packages one
# level down, so ComfyUI would not see them where the others sit.
for parent in BatchnodeI9 savezipi9; do
  dir="${CUSTOM_NODES}/${parent}"
  [ -d "$dir" ] || continue
  if [ ! -f "${dir}/__init__.py" ]; then
    for sub in "$dir"/*/; do
      [ -f "${sub}__init__.py" ] || continue
      echo "[nodes] hoisting $(basename "$sub") out of ${parent}"
      mv "$sub" "${CUSTOM_NODES}/$(basename "$sub")"
    done
    # rm -rf, not rmdir: a leftover README keeps the directory alive, and
    # ComfyUI then tries to import a package with no __init__.py and logs a
    # FileNotFoundError traceback on every boot.
    rm -rf "$dir"
  fi
done

echo "[nodes] installing package requirements"
for req in "${CUSTOM_NODES}"/*/requirements.txt; do
  [ -f "$req" ] || continue
  # Never let a node package pull its own torch: it would silently replace the
  # cu128 build from the base image with a CPU wheel, and every render would
  # then silently run on the CPU.
  filtered="$(mktemp)"
  grep -viE '^\s*(torch|torchvision|torchaudio|torchsde|numpy|transformers|tokenizers|protobuf|opencv-python)\s*([<=>!].*)?$' \
    "$req" > "$filtered" || true
  pip install --no-cache-dir --prefer-binary -c "$CONSTRAINTS" -r "$filtered" || {
    echo "[nodes] WARNING: requirements failed for $(dirname "$req")"; }
  rm -f "$filtered"
done

echo "[nodes] done"
