#!/usr/bin/env bash
# Fetch one model into the image. Called once per model so each lands in its own
# layer: a worker that already has the checkpoint does not re-pull it because a
# LoRA changed, and a code-only rebuild re-pulls neither.
#
#   fetch_model.sh hf     <repo> <remote-path> <out>
#   fetch_model.sh civit  <url>                <out>
#   fetch_model.sh url    <url>                <out>   (no token, public link)
#
# Tokens come from BuildKit secrets (/run/secrets/...), never from build args —
# an ARG is recorded in the layer history of the stage that declares it. They are
# also never put in a URL: curl prints the URL in some error paths, and a build
# log is a bad place for a credential even a masked one.
set -euo pipefail

kind="${1:?hf|civit|url}"
out="${!#}"
mkdir -p "$(dirname "$out")"

# Two callers, two ways of holding a token: at container start it is an ordinary
# env var from the RunPod endpoint, and in a build it would be a BuildKit secret
# file. Check the file first, fall back to the environment.
read_secret() {
  local path="${SECRETS_DIR:-/run/secrets}/$1"
  if [ -f "$path" ]; then
    tr -d '\r\n' < "$path"
    return
  fi
  # civitai_token -> CIVITAI_TOKEN
  local name
  name=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
  printf '%s' "$(eval "printf '%s' \"\${${name}:-}\"" | tr -d '\r\n')"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_MODEL="${HERE}/check_model.py"

# 0 complete safetensors, 2 truncated, 1 not safetensors. See check_model.py —
# shared with fetch_models.sh so the two cannot disagree on what a model is.
safetensors_state() {
  python3 "$CHECK_MODEL" "$1"
}

# A stalled transfer is not a slow one. Bounding a 13G download with a wall
# clock would kill legitimate fetches on a bad night, so the handshake gets a
# deadline and the transfer gets a floor: give up only when nothing is moving.
# --speed-limit/--speed-time abort below 1KB/s sustained for a minute; aria2
# spells the same thing --lowest-speed-limit.
#
# Without these a download does not fail, it hangs: Civitai accepted the
# connection, never answered, and the boot sat there until RunPod reaped the
# container ~8 minutes later — having thrown away 52G it had already fetched.
CONNECT_TIMEOUT=30
STALL_BYTES=1024
STALL_SECONDS=60
ARIA_STALL=(--lowest-speed-limit=1K --connect-timeout=30 --timeout=60)

# A build that half-downloads a checkpoint must not produce a working image with
# a truncated model inside it — that surfaces as a cryptic load error at render
# time, on the Sunday batch, with nothing in the build log to explain it.
verify() {
  local http="${1:-?}"
  local bytes=0
  [ -f "$out" ] && bytes=$(stat -c %s "$out")

  local state=1
  if [ "$bytes" -gt 8 ]; then
    state=0
    safetensors_state "$out" || state=$?
  fi

  # A size floor was not a harmless approximation of this: fedor_bypass is 1040
  # bytes because it carries a single [1, 12] tensor, and a >1MiB test called it
  # an error page, failed the fetch and crash-looped the worker on every cold
  # start. Anything we cannot parse is still judged on size, so a format this
  # does not understand cannot start failing a boot that works today.
  local truncated=""
  case "$state" in
    0) echo "[fetch] ok: $out ($(du -h "$out" | cut -f1), HTTP ${http})"
       return 0 ;;
    2) truncated=1 ;;
    *) if [ "$bytes" -gt 1048576 ] && [ "$(head -c 1 "$out")" != "<" ]; then
         echo "[fetch] ok: $out ($(du -h "$out" | cut -f1), HTTP ${http})"
         return 0
       fi ;;
  esac

  echo "[fetch] ERROR: ${out} is not a model."
  echo "        HTTP status : ${http}"
  echo "        bytes        : ${bytes}"
  if [ -n "$truncated" ]; then
    echo "        -> this is safetensors, but the file is shorter than its own"
    echo "           header says it should be: the download was cut off part"
    echo "           way. Retrying usually fixes it."
  else
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
  fi
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
      echo "[fetch] ERROR: no Civitai token. This file is gated."
      echo "        Set CIVITAI_TOKEN as an environment variable on the RunPod"
      echo "        endpoint (Settings -> Environment Variables). Get one from"
      echo "        civitai.com -> Account settings -> API Keys."
      exit 1
    fi

    # aria2 with 16 connections, where it exists. The checkpoint is the largest
    # single thing a cold start pulls and one stream does not saturate the link;
    # this is the difference between minutes and tens of minutes.
    #
    # Civitai answers with a redirect to a signed CDN URL, and the CDN rejects a
    # request that still carries the Civitai Authorization header — which is why
    # handing aria2 the original URL plus the header failed and fell back to a
    # single curl stream. So resolve the redirect first and give aria2 the
    # signed URL with no header at all.
    #
    # A *range* GET, not a HEAD. The signature is issued for the method that
    # asked for it: a URL recovered from a HEAD chain serves a web page, which
    # is exactly how this broke once before. `-r 0-0` is a real GET that
    # transfers one byte.
    #
    # This transfers one byte, so it has no business taking longer than a
    # moment — and it is exactly where the boot used to hang, with no timeout
    # of any kind on it. Bounded hard, and retried, so a Civitai hiccup costs
    # seconds and falls through to curl rather than costing the container.
    signed=""
    if command -v aria2c >/dev/null 2>&1; then
      signed=$(curl -sS -L -o /dev/null -r 0-0 \
                 --connect-timeout "$CONNECT_TIMEOUT" --max-time 60 \
                 --retry 3 --retry-delay 2 --retry-all-errors \
                 -H "Authorization: Bearer ${token}" \
                 -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
                 -w '%{url_effective}' "$url" 2>/dev/null) || signed=""
    fi

    if [ -n "$signed" ] && [ "$signed" != "$url" ]; then
      echo "[fetch] resolved to $(printf '%s' "$signed" | sed 's#\(https\?://[^/]*\)/.*#\1/…#')"
      if aria2c -x 16 -s 16 -k 1M --split=16 --min-split-size=1M \
                --max-tries=5 --retry-wait=3 "${ARIA_STALL[@]}" \
                --allow-overwrite=true --file-allocation=none \
                --console-log-level=warn --summary-interval=0 \
                -d "$(dirname "$out")" -o "$(basename "$out")" "$signed"; then
        verify 200
        exit 0
      fi
      echo "[fetch] aria2 failed on the signed URL, falling back to curl"
    elif command -v aria2c >/dev/null 2>&1; then
      echo "[fetch] no redirect to resolve, using curl"
    fi

    # Fallback: one curl that follows its own redirect. Slower (single stream)
    # but it needs nothing resolved in advance, and curl drops the auth header
    # on the cross-host hop by itself.
    http=$(curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
             --connect-timeout "$CONNECT_TIMEOUT" \
             --speed-limit "$STALL_BYTES" --speed-time "$STALL_SECONDS" \
             -H "Authorization: Bearer ${token}" \
             -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
             -w '%{http_code}' -o "$out" "$url") || http="000"
    verify "$http"
    ;;
  url)
    # A plain public link — no token, no signed-redirect dance. Used for weights
    # that live somewhere other than HF or Civitai, currently the Klein style
    # LoRA on Dropbox.
    #
    # verify() is what makes this safe to point at a file-sharing host: those
    # serve an HTML preview page rather than a 404 when a link stops working,
    # and verify rejects a body that opens with "<" instead of writing a
    # plausible-looking file that fails at load time on the Sunday batch.
    url="$2"
    echo "[fetch] url $(basename "$out")"
    if command -v aria2c >/dev/null 2>&1; then
      if aria2c -x 16 -s 16 -k 1M --split=16 --min-split-size=1M \
                --max-tries=5 --retry-wait=3 "${ARIA_STALL[@]}" \
                --allow-overwrite=true --file-allocation=none \
                --console-log-level=warn --summary-interval=0 \
                -d "$(dirname "$out")" -o "$(basename "$out")" "$url"; then
        verify 200
        exit 0
      fi
      echo "[fetch] aria2 failed, falling back to curl"
    fi
    http=$(curl -sSL --retry 8 --retry-delay 3 --retry-all-errors \
             --connect-timeout "$CONNECT_TIMEOUT" \
             --speed-limit "$STALL_BYTES" --speed-time "$STALL_SECONDS" \
             -w '%{http_code}' -o "$out" "$url") || http="000"
    verify "$http"
    ;;
  *)
    echo "[fetch] unknown kind: $kind"; exit 2 ;;
esac
