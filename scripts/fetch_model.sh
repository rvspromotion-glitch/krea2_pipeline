#!/usr/bin/env bash
# Fetch one model into the image. Called once per model so each lands in its own
# layer: a worker that already has the checkpoint does not re-pull it because a
# LoRA changed, and a code-only rebuild re-pulls neither.
#
#   fetch_model.sh hf     <repo> <remote-path> <out>
#   fetch_model.sh civit  <url>                <out>
#
# Tokens come from BuildKit secrets (/run/secrets/...), never from build args —
# an ARG is recorded in the layer history of the stage that declares it.
set -euo pipefail

kind="${1:?hf|civit}"
out="${!#}"
mkdir -p "$(dirname "$out")"

read_secret() {
  local path="/run/secrets/$1"
  [ -f "$path" ] && tr -d '\r\n' < "$path" || true
}

# A build that half-downloads a checkpoint must not produce a working image with
# a truncated model inside it — that surfaces as a cryptic load error at render
# time, on the Sunday batch, with nothing in the build log to explain it.
verify() {
  [ -s "$out" ] || { echo "[fetch] ERROR: $out is empty"; rm -f "$out"; exit 1; }

  # An unauthenticated or gated fetch answers with an HTML login page, which
  # would sit in the image looking exactly like a model.
  if [ "$(head -c 1 "$out")" = "<" ]; then
    echo "[fetch] ERROR: $out is a web page, not a model — the download was"
    echo "        rejected (gated file, or a missing/expired token)."
    head -c 300 "$out"; echo
    rm -f "$out"; exit 1
  fi

  local bytes; bytes=$(stat -c %s "$out")
  if [ "$bytes" -lt 1048576 ]; then
    echo "[fetch] ERROR: $out is only ${bytes} bytes — that is an error page,"
    echo "        not a model."
    rm -f "$out"; exit 1
  fi
  echo "[fetch] ok: $out ($(du -h "$out" | cut -f1))"
}

case "$kind" in
  hf)
    repo="$2"; remote="$3"
    echo "[fetch] hf ${repo}/${remote}"
    # The Python API rather than the CLI: the command was renamed from
    # `huggingface-cli download` to `hf download` and the old one is on its way
    # out, whereas hf_hub_download has been stable throughout. hf_transfer is a
    # rust downloader that saturates the link, which matters at these sizes.
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_TOKEN="$(read_secret hf_token)" \
    python3 - "$repo" "$remote" "$out" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

repo, remote, out = sys.argv[1:4]
token = os.environ.get("HF_TOKEN") or None
path = hf_hub_download(repo_id=repo, filename=remote, token=token)
# copy, not move: the download lands in the hub cache as a symlink target.
shutil.copyfile(path, out)
PY
    ;;
  civit)
    url="$2"
    echo "[fetch] civitai $(basename "$out")"
    token="$(read_secret civitai_token)"
    dl="$url"
    if [ -n "$token" ]; then
      # Civitai redirects to a signed CDN URL. Resolve it here so the token is
      # never sent on to the CDN host.
      resolved=$(curl -sSL -I -H "Authorization: Bearer ${token}" "$url" \
        | grep -i '^location:' | tail -1 | sed 's/^location: //i' | tr -d '\r\n') || resolved=""
      [ -n "$resolved" ] && dl="$resolved"
    fi
    curl -fL --retry 8 --retry-delay 3 --retry-all-errors -C - \
      -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
      -o "$out" "$dl"
    ;;
  *)
    echo "[fetch] unknown kind: $kind"; exit 2 ;;
esac

verify
