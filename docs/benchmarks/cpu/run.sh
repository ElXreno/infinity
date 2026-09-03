#!/usr/bin/env bash
# Usage: run.sh <image> <label> [extra infinity flags...]
#
# Starts one infinity container with an embedder and a reranker, waits for /health, runs the
# workload against it and prints the anonymous RSS of the server process at the end.
#
#   WORKLOAD          mixed | titles | embed-reference | score-reference (default: mixed)
#   BENCH_OUT         directory for reference dumps (default: current directory)
#   BENCH_CPUS        --cpus for the container (default: 8)
#   BENCH_MEMORY      --memory for the container (default: 24g)
#   BENCH_PORT        host port (default: 17997)
#   HF_CACHE_VOLUME   podman volume mounted as HF_HOME (default: infinity-hf-cache)
#   EMBED_MODEL, EMBED_REVISION, EMBED_BATCH, RERANK_MODEL, RERANK_REVISION, RERANK_BATCH
#   EXTRA_PODMAN_ARGS extra arguments for `podman run`, split on whitespace
#   PYTHON            interpreter for the workload, may carry arguments (default: python3)
set -u
IMAGE="$1"
LABEL="$2"
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PORT="${BENCH_PORT:-17997}"
NAME="${BENCH_CONTAINER:-infinity-bench}"
EXTRA=()
[ -n "${EXTRA_PODMAN_ARGS:-}" ] && read -ra EXTRA <<<"$EXTRA_PODMAN_ARGS"

podman rm -f "$NAME" >/dev/null 2>&1
podman run -d --name "$NAME" --cpus "${BENCH_CPUS:-8}" --memory "${BENCH_MEMORY:-24g}" \
  -p "127.0.0.1:${PORT}:7997" "${EXTRA[@]}" \
  -v "${HF_CACHE_VOLUME:-infinity-hf-cache}:/data" -e HF_HOME=/data -e HF_HUB_DISABLE_XET=1 \
  -e INFINITY_ONNX_DISABLE_OPTIMIZE="${INFINITY_ONNX_DISABLE_OPTIMIZE:-true}" \
  -e TOKENIZERS_PARALLELISM=true -e INFINITY_PORT=7997 \
  "$IMAGE" v2 \
  --model-id "${EMBED_MODEL:-ElXreno/LaBSE-en-ru-onnx}" \
  --model-id "${RERANK_MODEL:-keisuke-miyako/bge-reranker-v2-m3-onnx-f32}" \
  --revision "${EMBED_REVISION:-main}" --revision "${RERANK_REVISION:-main}" \
  --served-model-name embed --served-model-name rerank \
  --batch-size "${EMBED_BATCH:-16}" --batch-size "${RERANK_BATCH:-4}" \
  --engine optimum --engine optimum --pooling-method cls \
  --log-level warning "$@" >/dev/null || {
  echo "podman run failed"
  exit 1
}

for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)
  [ "$code" = "200" ] && break
  if [ "$(podman inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "== $LABEL: container died during startup"
    podman logs --tail 15 "$NAME" 2>&1 | cut -c1-200
    podman rm -f "$NAME" >/dev/null 2>&1
    exit 1
  fi
  sleep 5
done
echo "== $LABEL: /health $code after $((i * 5))s"
sleep 5

read -ra PY <<<"${PYTHON:-python3}"
"${PY[@]}" "$HERE/workload.py" --port "$PORT" --label "$LABEL" \
  --out "${BENCH_OUT:-$PWD}" "${WORKLOAD:-mixed}"
echo "== $LABEL: RSS $(podman exec "$NAME" awk '/^Anonymous:/ {print $2/1024 " MiB anon"}' /proc/1/smaps_rollup)"
podman rm -f "$NAME" >/dev/null 2>&1
