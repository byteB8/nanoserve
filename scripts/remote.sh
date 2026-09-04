#!/usr/bin/env bash
# Push the working tree to the GPU box, run a benchmark there, pull results back.
#
# Code stays local and authoritative; the server is treated as a disposable
# executor. Nothing is committed to get an experiment run, so the git history
# stays about the engine rather than about debugging cycles.
#
#   ./scripts/remote.sh setup                     # create remote venv, install deps
#   ./scripts/remote.sh push                      # sync source only
#   ./scripts/remote.sh bench all --device cuda   # sync, run, fetch results/
#   ./scripts/remote.sh shell                     # interactive session on the box
#
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE="scripts/remote.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found. Copy scripts/remote.env.example and fill it in." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

SSH=(ssh -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}")
TARGET="${REMOTE_USER}@${REMOTE_HOST}"

push() {
  echo ">> syncing source to ${TARGET}:${REMOTE_DIR}"
  "${SSH[@]}" "mkdir -p ${REMOTE_DIR}"
  rsync -az --delete \
    -e "ssh -p ${REMOTE_PORT}" \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.venv' --exclude 'results/' --exclude 'scripts/remote.env' \
    ./ "${TARGET}:${REMOTE_DIR}/"
}

fetch() {
  echo ">> fetching results"
  mkdir -p results
  rsync -az -e "ssh -p ${REMOTE_PORT}" "${TARGET}:${REMOTE_DIR}/results/" ./results/ || true
}

case "${1:-}" in
  setup)
    push
    echo ">> creating remote venv and installing dependencies"
    "${SSH[@]}" "cd ${REMOTE_DIR} && ${REMOTE_PYTHON} -m venv .venv && \
      .venv/bin/pip install -q --upgrade pip && \
      .venv/bin/pip install -q -r requirements.txt && \
      .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"
    ;;
  push)
    push
    ;;
  bench)
    shift
    push
    echo ">> running: nanoserve.bench $*"
    "${SSH[@]}" "cd ${REMOTE_DIR} && mkdir -p results && \
      .venv/bin/python -m nanoserve.bench $* --out results/remote.md"
    fetch
    echo ">> results/remote.md updated"
    ;;
  shell)
    exec "${SSH[@]}"
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
