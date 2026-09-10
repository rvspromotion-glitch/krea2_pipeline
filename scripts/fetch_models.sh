#!/usr/bin/env bash
# Fetch every model in models.txt into MODELS_DIR. Run at container start.
#
# The weights are not in the image: ~39GB of models on top of torch does not
# build on a hosted GitHub runner. So the image carries ComfyUI, the nodes and
# the code, and the worker pulls the weights once on cold start.
#
# For this workload that is the right side of the trade. The weekly render is a
# sequential batch of forty-odd jobs on one warm worker, so the download is paid
# once for the whole run, against three to four hours of rendering.
#
# Downloads run concurrently — five files from three hosts, and the link is
# nowhere near saturated by one of them. Each one's output is captured and
# printed when it finishes, so five interleaved progress bars do not become one
# unreadable log — but that alone means total silence for minutes, and the last
# line on screen is whichever download *started* last, which reads as "stuck on
# a 110MB LoRA" when it is really the 12GB checkpoint still going. Hence the
# ticker: one compact line every PROGRESS_EVERY seconds with what is on disk.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="${MODELS_LIST:-${HERE}/../models.txt}"
MODELS_DIR="${MODELS_DIR:-/comfyui/models}"
FETCH_ONE="${HERE}/fetch_model.sh"
CHECK_MODEL="${HERE}/check_model.py"
MIN_BYTES=1048576

# Enough to overlap the slow host with the fast ones without hammering either.
MAX_PARALLEL="${MODEL_FETCH_PARALLEL:-4}"

# How often the ticker reports. Often enough to look alive, rare enough not
# to bury the real messages.
PROGRESS_EVERY="${MODEL_FETCH_PROGRESS_EVERY:-15}"

log() { echo "[models] $*"; }

# Present means complete, not merely large. A size floor got this wrong in both
# directions: fedor_bypass is 1040 bytes of real model, so it never counted as
# present and was re-fetched from Civitai on every boot — and with no volume
# attached, every avoidable Civitai round-trip is another chance to hang. A
# truncated file reports state 2 and is re-fetched rather than trusted, which is
# what lets a cut-off download heal itself on the next start.
have_it() {
  local path="$1"
  [ -f "$path" ] || return 1
  local state=0
  python3 "$CHECK_MODEL" "$path" || state=$?
  case "$state" in
    0) return 0 ;;
    2) return 1 ;;
    *) [ "$(stat -c %s "$path")" -ge "$MIN_BYTES" ] ;;
  esac
}

if [ ! -f "$LIST" ]; then
  echo "[models] FATAL: no model list at ${LIST}" >&2
  exit 1
fi

started=$(date +%s)
declare -a pids=() names=() logs=() wanted=()
queued=0
present=0

while read -r kind a b c; do
  case "$kind" in ""|\#*) continue ;; esac

  case "$kind" in
    hf)    repo="$a"; remote="$b"; rel="$c" ;;
    civit) url="$a";  rel="$b" ;;
    url)   url="$a";  rel="$b" ;;
    *)     echo "[models] FATAL: unknown kind '${kind}' in ${LIST}" >&2; exit 1 ;;
  esac

  dest="${MODELS_DIR}/${rel}"
  name="$(basename "$rel")"

  # Skip-if-present is what makes a restarted container, or a volume if one is
  # ever attached, cost nothing.
  if have_it "$dest"; then
    log "have ${name} ($(du -h "$dest" | cut -f1))"
    present=$((present + 1))
    continue
  fi

  # Bound concurrency: wait for a slot before starting another.
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 1; done

  logfile="$(mktemp)"
  if [ "$kind" = "hf" ]; then
    "$FETCH_ONE" hf "$repo" "$remote" "$dest" >"$logfile" 2>&1 &
  else
    "$FETCH_ONE" "$kind" "$url" "$dest" >"$logfile" 2>&1 &
  fi
  pids+=($!); names+=("$name"); logs+=("$logfile"); wanted+=("$dest")
  queued=$((queued + 1))
  log "fetching ${name}…"
done < "$LIST"

# Ticker, for as long as anything is in flight. Sizes come off the filesystem
# rather than from the downloaders, so it works the same for aria2 and curl.
# Sleeps in one-second steps rather than one long one. `kill` on the subshell
# does not reach an external `sleep` it is blocked in, so that orphan lives on
# holding this script's stdout open — which stalls anything reading our output
# for the rest of the interval. A flag file it can notice quickly avoids that.
progress_ticker() {
  local waited=0
  while [ -f "$TICK_FLAG" ]; do
    sleep 1
    waited=$((waited + 1))
    [ "$waited" -lt "$PROGRESS_EVERY" ] && continue
    waited=0
    line=""
    for dest in "${wanted[@]}"; do
      short="$(basename "$dest")"; short="${short%.safetensors}"
      if [ -f "$dest" ]; then
        line="${line}${short} $(du -h "$dest" 2>/dev/null | cut -f1)  "
      else
        line="${line}${short} …  "
      fi
    done
    echo "[models] $(( $(date +%s) - started ))s  ${line}"
  done
}

TICK_FLAG=""
if [ "${#pids[@]}" -gt 0 ]; then
  TICK_FLAG="$(mktemp)"
  progress_ticker &
  TICKER=$!
  # Never outlive the fetch, however this script exits.
  trap 'rm -f "$TICK_FLAG"' EXIT
fi

# Reap in the order they finish, not the order they were listed. `wait` on one
# pid at a time blocks the whole report behind whichever file is slowest: when
# Realism_engine hung, the four downloads that had already finished after it in
# the list could not print, so the log showed ten of fifteen done and read as
# five simultaneous failures instead of one stuck file. `wait -n` returns as
# each child exits, so the last name still outstanding is the one to blame.
declare -A pending=()
for i in "${!pids[@]}"; do pending["${pids[$i]}"]="$i"; done

# `wait -n -p VAR <pids>` needs bash 5.1. Listing the pids matters as much as
# -p does: a bare `wait -n` would also reap the ticker, which is a child too.
# Where it is unavailable the old list order is used — that is only a reporting
# nicety, correctness never depended on the order these are reaped in.
by_completion=0
if [ "${BASH_VERSINFO[0]}" -gt 5 ] ||
   { [ "${BASH_VERSINFO[0]}" -eq 5 ] && [ "${BASH_VERSINFO[1]}" -ge 1 ]; }; then
  by_completion=1
fi

failed=0
while [ "${#pending[@]}" -gt 0 ]; do
  finished=""
  code=0
  if [ "$by_completion" = 1 ]; then
    wait -n -p finished "${!pending[@]}" || code=$?
  fi
  if [ -z "$finished" ] || [ -z "${pending[$finished]:-}" ]; then
    for pid in "${!pending[@]}"; do finished="$pid"; break; done
    code=0
    wait "$finished" || code=$?
  fi

  i="${pending[$finished]}"
  unset 'pending[$finished]'
  if [ "$code" -eq 0 ]; then
    tail -n 2 "${logs[$i]}" | sed 's/^/[models] /'
  else
    failed=$((failed + 1))
    echo "[models] FAILED: ${names[$i]}" >&2
    sed 's/^/[models]   /' "${logs[$i]}" >&2
  fi
  rm -f "${logs[$i]}"
done

if [ -n "$TICK_FLAG" ]; then
  rm -f "$TICK_FLAG"
  wait "$TICKER" 2>/dev/null || true
  trap - EXIT
fi

elapsed=$(( $(date +%s) - started ))

if [ "$failed" -gt 0 ]; then
  echo "[models] FATAL: ${failed} of ${queued} download(s) failed after ${elapsed}s." >&2
  echo "[models]        The worker cannot render without them, so it is stopping" >&2
  echo "[models]        rather than accepting jobs that would all fail." >&2
  echo "[models]        A 401 above means CIVITAI_TOKEN is missing or wrong on the" >&2
  echo "[models]        RunPod endpoint (Settings -> Environment Variables)." >&2
  exit 1
fi

log "ready — ${present} already present, ${queued} fetched in ${elapsed}s"
du -sh "$MODELS_DIR" 2>/dev/null | sed 's/^/[models] total /'
