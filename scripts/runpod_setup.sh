#!/usr/bin/env bash
# RunPod setup for pixallm SFT training.
#
# Usage (from repo root after `git clone`):
#   bash scripts/runpod_setup.sh               # env + deps only
#   bash scripts/runpod_setup.sh --prepare     # also build train_v1.jsonl
#   bash scripts/runpod_setup.sh --smoke       # also run a 50-step SFT smoke test
#
# Expects a .env file in the repo root with:
#   HF_TOKEN=...          # optional (nouns dataset is CC0/public)
#   WANDB_API_KEY=...     # optional, only if you pass --report-to wandb to sft

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PREPARE_DATA=0
RUN_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --prepare) PREPARE_DATA=1 ;;
    --smoke)   RUN_SMOKE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# 1. Point HF cache at the RunPod network volume so model weights survive restarts.
if [[ -d /workspace ]]; then
  export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/.hf_cache/datasets}"
  mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
  log "HF cache -> $HF_HOME"
else
  log "no /workspace volume detected; HF cache stays at \$HOME default"
fi

# 2. Load secrets from .env if present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  log ".env loaded"
else
  log ".env not found — copy .env.example to .env and fill in keys if you need HF/wandb"
fi

# 3. Install uv if the image doesn't ship it.
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv $(uv --version)"

# 4. Sync the locked dependency set.
log "uv sync --frozen"
uv sync --frozen

# 5. Linux NCCL workaround — some CUDA base images ship an older NCCL that
#    clashes with torch 2.11. Prepend the bundled one for any command we run.
NCCL_LIB_DIR="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
if [[ -d "$NCCL_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  log "LD_LIBRARY_PATH prefixed with bundled NCCL"
fi

# 6. CUDA / torch sanity check.
log "torch / cuda check"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 7. wandb login (only if key present).
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  log "wandb login"
  uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null
fi

# 8. Training data — optional because the nouns download + quantization takes a while.
if [[ "$PREPARE_DATA" == "1" ]]; then
  if [[ -s data/processed/train_v1.jsonl ]]; then
    log "train_v1.jsonl already exists ($(wc -l < data/processed/train_v1.jsonl) lines); skipping prepare"
  else
    log "preparing data/processed/train_v1.jsonl (3000 samples)"
    uv run python scripts/prepare_data.py --limit 3000 --output data/processed/train_v1.jsonl
  fi
fi

# 9. Optional smoke test so you catch config / chat-template errors before the long run.
if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ ! -s data/processed/train_debug_50.jsonl ]]; then
    log "building train_debug_50.jsonl for smoke test"
    uv run python scripts/prepare_data.py --limit 50 --output data/processed/train_debug_50.jsonl
  fi
  log "50-step SFT smoke test (no wandb, throwaway output dir)"
  uv run python -m pixallm.train.sft \
    --train-file data/processed/train_debug_50.jsonl \
    --output-dir checkpoints/smoke \
    --max-steps 50 \
    --save-steps 50 \
    --logging-steps 5 \
    --report-to none
fi

log "done. next: uv run python -m pixallm.train.sft --report-to wandb"
