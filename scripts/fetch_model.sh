#!/usr/bin/env bash
# Fetch one model into the image. Called once per model so each lands in its own
# layer: a worker that already has the checkpoint does not re-pull it because a
# LoRA changed, and a code-only rebuild re-pulls neither.
#
#   fetch_model.sh hf     <repo> <remote-path> <out>
#   fetch_model.sh civit  <url>                <out>
#
# Tokens come from BuildKit secrets (/run/secrets/...), never from build args —
# an ARG is recorded in the layer history of the stage that declares it. They are
# also never put in a URL: curl prints the URL in some error paths, and a build
# log is a bad place for a credential even a masked one.
set -euo pipefail

kind="${1:?hf|civit}"
out="${!#}"
mkdir -p "$(dirname "$out")"

# SECRETS_DIR is overridable so this is testable off a real BuildKit mount; the
# build never sets it.
read_secret() {
  local path="${SECRETS_DIR:-/run/secrets}/$1"
  [ -f "$path" ] && tr -d '\r\n' < "$path" || true
}

# A build that half-downloads a checkpoint must not produce a working image with
# a truncated model inside it — that surfaces as a cryptic load error at render
# time, on the Sunday batch, with nothing in the build log to explain it.
verify() {
  local http="${1:-?}"
  local bytes=0
  [ -f "$out" ] && bytes=$(stat -c %s "$out")

  if [ "$bytes" -gt 1048576 ] && [ "$(head -c 1 "$out")" != "<" ]; then
    echo "[fetch] ok: $out ($(du -h "$out" | cut -f1), HTTP ${http})"
    return 0
  fi

  echo "[fetch] ERROR: ${out} is not a model."
  echo "        HTTP status : ${http}"
  echo "        bytes        : ${bytes}"
  echo "        first 400 bytes of what the server actually sent:"
  head -c 400 "$out" 2>/dev/null | sed 's/^/        | /'
  echo
  case "$http" in
    401|403) echo "        -> the token was rejected. Check the CIVITAI_TOKEN /"
             echo "           HF_TOKEN repo secret, and that the account can"
             echo "           download this file." ;;
    404)     echo "        -> the model id or fileId is wrong, or the file was"
             echo "           taken down." ;;
    200)     echo "        -> HTTP 200 with a web page in the body. The URL"
             echo "           resolved to a page rather than a file." ;;
  esac
  rm -f "$out"
  exit 1
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
    verify 200
    ;;
  civit)
    url="$2"
    echo "[fetch] civitai $(basename "$out")"
    token="$(read_secret civitai_token)"
    if [ -z "$token" ]; then
      echo "[fetch] ERROR: the civitai_token build secret is empty."
      echo "        This file is gated. Add a CIVITAI_TOKEN repo secret"
      echo "        (Settings -> Secrets and variables -> Actions)."
      exit 1
    fi

    # One request, and let curl follow the redirect itself.
    #
    # The previous version pre-resolved the redirect with `curl -I` so the token
    # would not travel to the CDN. That is what broke this build: Civitai's
    # signed CDN URL is issued for a GET, and the URL recovered from a HEAD
    # chain served an HTML page, which then got downloaded and written out as a
    # 9KB "checkpoint".
    #
    # It also bought nothing. curl already drops the Authorization header on a
    # cross-host redirect, which is exactly the property being hand-rolled, and
    # the CDN URL it lands on is signed and needs no credential.
    http=$(curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
             -H "Authorization: Bearer ${token}" \
             -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
             -w '%{http_code}' -o "$out" "$url") || http="000"
    verify "$http"
    ;;
  *)
    echo "[fetch] unknown kind: $kind"; exit 2 ;;
esac
