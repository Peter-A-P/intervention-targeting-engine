#!/usr/bin/env bash
# Time TabFM on a rented GPU against the CPU timings already committed.
#
# Run on a fresh Linux GPU machine (an A100 80GB was the one priced) from an empty
# directory. It installs the project, fetches the two small datasets it needs, and runs
# three timed jobs into results/bases-pilot/, which is not the committed results directory:
#
#   1. ACIC seed 11, all five meta-learners, float32. The same job took 2 h 24 min of
#      TabFM time on an 8-thread laptop CPU, so the ratio is the speedup.
#   2. The same in bfloat16, the precision TabFM was designed for, to see what it buys
#      on hardware that supports it natively.
#   3. Hillstrom seed 11, T-learner only, in bfloat16 at the full 16,384-row context: the first job
#      that reaches the context cap, and the one the cost of the larger sets scales from.
#
# Each job has a time limit, so a slower GPU than expected costs a known amount.
# Copy results/bases-pilot/ and pilot.log back when it finishes.
set -euo pipefail

BRANCH="${BRANCH:-tabfm-base-learner}"
LIMIT="${LIMIT:-45m}"
# A subset, in order, when time is short: JOBS="acic-bfloat16 hillstrom-t".

nvidia-smi --query-gpu=name,memory.total --format=csv | tee pilot.log

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --branch "$BRANCH" --depth 1 \
  https://github.com/Peter-A-P/intervention-targeting-engine.git itx
cd itx
uv sync --extra learners --extra foundation
# The locked torch wheel is built for CUDA 13, which needs a recent driver. On a machine
# with an older one, swap in the same torch built for CUDA 12.6 and stop uv re-syncing it.
if ! uv run python -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
  version=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
  uv pip install --reinstall "torch==$version" --index-url https://download.pytorch.org/whl/cu126
  export UV_NO_SYNC=1
fi
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"
uv run itx data pull hillstrom acic-x acic-zymu-1

run() {
  local name="$1"; shift
  local start=$SECONDS
  echo "== $name" | tee -a ../pilot.log
  timeout "$LIMIT" uv run itx bases --results-dir results/bases-pilot/"$name" \
    --readme /dev/null --device cuda "$@" 2>&1 | tee -a ../pilot.log || true
  echo "== $name: $((SECONDS - start)) s wall-clock" | tee -a ../pilot.log
}

for job in ${JOBS:-acic-float32 acic-bfloat16 hillstrom-t}; do
  case "$job" in
    acic-float32) run "$job" --dataset acic --seeds 11 ;;
    acic-bfloat16) run "$job" --dataset acic --seeds 11 --bfloat16 ;;
    hillstrom-t) run "$job" --dataset hillstrom --seeds 11 --learners t-learner --bfloat16 ;;
    *) echo "unknown job $job" >&2; exit 2 ;;
  esac
done
