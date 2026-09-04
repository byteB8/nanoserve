#!/usr/bin/env bash
# Push the working tree to a GPU box, run a benchmark there, pull results back.
#
# Code stays local and authoritative; the server is treated as a disposable
# executor. Nothing needs committing to run an experiment, so the git history
# stays about the engine rather than about debugging cycles.
#
#   ./scripts/remote.sh setup                     # create remote venv, install deps
#   ./scripts/remote.sh push                      # sync source only
#   ./scripts/remote.sh gpus                      # who is using the GPUs right now
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

TARGET="${REMOTE_USER}@${REMOTE_HOST}"

# Pin a single identity. Offering every key in ~/.ssh gets the connection dropped
# by any server with a low MaxAuthTries long before the right key is tried.
SSH_OPTS=(-p "$REMOTE_PORT")
if [[ -n "${REMOTE_SSH_KEY:-}" ]]; then
  SSH_OPTS+=(-o IdentitiesOnly=yes -i "${REMOTE_SSH_KEY/#\~/$HOME}")
fi
SSH=(ssh "${SSH_OPTS[@]}" "$TARGET")
RSH="ssh ${SSH_OPTS[*]}"

# Prefix for any remote command that needs the toolchain: module init, then the
# GPU pin. Kept here rather than baked into each call site.
prelude() {
  local p=""
  [[ -n "${REMOTE_INIT:-}" ]] && p+="${REMOTE_INIT} && "
  [[ -n "${CUDA_DEVICES:-}" ]] && p+="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} && "
  printf '%s' "$p"
}

push() {
  echo ">> syncing source to ${TARGET}:${REMOTE_DIR}"
  "${SSH[@]}" "mkdir -p ${REMOTE_DIR}"
  rsync -az --delete -e "$RSH" \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.venv' --exclude 'results/' --exclude 'scripts/remote.env' \
    ./ "${TARGET}:${REMOTE_DIR}/"
}

fetch() {
  echo ">> fetching results"
  mkdir -p results
  rsync -az -e "$RSH" "${TARGET}:${REMOTE_DIR}/results/" ./results/ || true
}

case "${1:-}" in
  setup)
    push
    echo ">> creating remote venv and installing dependencies"
    "${SSH[@]}" "$(prelude) cd ${REMOTE_DIR} && ${REMOTE_PYTHON} -m venv .venv && \
      .venv/bin/pip install -q --upgrade pip && \
      .venv/bin/pip install -q -r requirements.txt && \
      .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"| cuda\", torch.cuda.is_available(), \"|\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu only\")'"
    ;;
  push)
    push
    ;;
  gpus)
    "${SSH[@]}" "nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv"
    ;;
  bench)
    shift
    tag="${TAG:-remote}"
    push
    echo ">> running: nanoserve.bench $*"
    "${SSH[@]}" "$(prelude) cd ${REMOTE_DIR} && mkdir -p results && \
      .venv/bin/python -m nanoserve.bench $* --out results/${tag}.md"
    fetch
    echo ">> results/${tag}.md updated"
    ;;
  shell)
    exec "${SSH[@]}"
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 1
    ;;
esac
