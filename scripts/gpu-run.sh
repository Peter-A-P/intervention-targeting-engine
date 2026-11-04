#!/usr/bin/env bash
# Run the base-learner comparison on a rented GPU, in the one setup every table reports.
#
# TabFM in bfloat16 on CUDA; LightGBM on the same machine's CPU, as everywhere else. Run
# from an empty directory on a fresh Linux GPU machine. IEEE-CIS cannot be downloaded
# without a Kaggle login, so ieee_train_transaction.csv is copied in beforehand to
# $IEEE_FILE; the other datasets come from their loaders, checksums verified.
#
# Datasets run in order, cheapest first, each with its own time limit, and every finished
# row is checkpointed, so an interrupted run loses at most one row: rerunning this script
# resumes. Results land in itx/results/bases/, the committed results directory, for
# copying back to the repository and redrawing the README with `itx bases --report-only`.
#
#   DATASETS="ihdp acic"  run a subset, in order
#   LIMIT=4h              per-dataset time limit
set -euo pipefail

BRANCH="${BRANCH:-tabfm-base-learner}"
LIMIT="${LIMIT:-4h}"
IEEE_FILE="${IEEE_FILE:-$PWD/ieee_train_transaction.csv}"
# Phase 1 at the committed five seeds; the three large sets at one seed (PLAN.md change 65).
DATASETS="${DATASETS:-ihdp acic hillstrom ieee-fraud lenta criteo}"

nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a run.log

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# A fresh clone drops the committed comparison files, which ran on the laptop CPU in
# float32: every row is rerun here, so the tables come from one setup. A rerun of this
# script finds the clone and resumes from its checkpoints instead.
if [ ! -d itx ]; then
  git clone --branch "$BRANCH" --depth 1 \
    https://github.com/Peter-A-P/intervention-targeting-engine.git itx
  rm -f itx/results/bases/*.json
fi
cd itx
# The locked torch wheel is built for CUDA 13, which needs a recent driver. On a machine
# with an older one, swap in the same torch built for CUDA 12.6 and stop uv re-syncing it.
# A rerun finds an environment that already sees the GPU and leaves it alone: syncing it
# again would put the CUDA 13 wheel back over the swap.
cuda_ok() { UV_NO_SYNC=1 uv run python -c "import torch, sys; sys.exit(not torch.cuda.is_available())" 2>/dev/null; }
if [ -d .venv ] && cuda_ok; then
  export UV_NO_SYNC=1
else
  uv sync --extra learners --extra foundation
fi
if ! cuda_ok; then
  version=$(grep -A1 '^name = "torch"$' uv.lock | sed -n 's/^version = "\(.*\)"$/\1/p')
  uv pip install --reinstall "torch==$version" --index-url https://download.pytorch.org/whl/cu126
  export UV_NO_SYNC=1
fi
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"

for dataset in $DATASETS; do
  seeds=""
  case "$dataset" in
    ihdp) uv run itx data pull ihdp-train ihdp-test ;;
    acic) uv run itx data pull acic-x acic-zymu-1 ;;
    hillstrom) uv run itx data pull hillstrom ;;
    lenta) uv run itx data pull lenta; seeds="--seeds 11" ;;
    criteo) uv run itx data pull criteo; seeds="--seeds 11" ;;
    ieee-fraud)
      mkdir -p data/raw
      [ -f data/raw/ieee_train_transaction.csv ] || cp "$IEEE_FILE" data/raw/
      seeds="--seeds 11" ;;
    *) echo "unknown dataset $dataset" >&2; exit 2 ;;
  esac
  start=$SECONDS
  echo "== $dataset" | tee -a ../run.log
  # shellcheck disable=SC2086
  timeout "$LIMIT" uv run itx bases --dataset "$dataset" $seeds --resume \
    --device cuda --bfloat16 --readme /dev/null 2>&1 | tee -a ../run.log || true
  echo "== $dataset: $((SECONDS - start)) s wall-clock" | tee -a ../run.log
done
echo RUN-FINISHED | tee -a ../run.log
