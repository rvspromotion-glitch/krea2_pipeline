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
  # GitHub's tarball endpoint does not resolve `HEAD` to the default branch:
  # for rgthree it served a 2023 `dev` branch that cannot load on this ComfyUI.
  if [ "$ref" = "HEAD" ]; then
    echo "[nodes] ERROR: ${name} has ref HEAD; pin a commit or name the branch" >&2
    exit 1
  fi
  echo "[nodes] ${name} <- ${repo}@${ref}"
  tmp="$(mktemp --suffix=.tar.gz)"
  curl -fSL --retry 5 --retry-delay 3 -o "$tmp" \
    "https://codeload.github.com/${repo}/tar.gz/${ref}"
  mkdir -p "$dest"
  # The tarball's top directory is <repo>-<full sha>: the commit actually
  # installed, which `HEAD` alone never tells you when a render misbehaves.
  top="$(tar -tzf "$tmp" | head -1)" || true
  echo "[nodes] ${name} is ${top%/}"
  if [[ "$ref" =~ ^[0-9a-f]{40}$ ]] && [[ "${top%/}" != *"-${ref}" ]]; then
    echo "[nodes] ERROR: ${name} pinned to ${ref} but the download is ${top%/}" >&2
    exit 1
  fi
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
  # opencv likewise: several packages ask for plain opencv-python, which
  # replaces the contrib build and takes cv2.ximgproc with it.
  grep -viE '^\s*(torch|torchvision|torchaudio|torchsde|numpy|transformers|tokenizers|protobuf|opencv-python|opencv-python-headless|opencv-contrib-python)\s*([<=>!].*)?$' \
    "$req" > "$filtered" || true
  pip install --no-cache-dir --prefer-binary -c "$CONSTRAINTS" -r "$filtered" || {
    echo "[nodes] WARNING: requirements failed for $(dirname "$req")"; }
  rm -f "$filtered"
done

echo "[nodes] done"
