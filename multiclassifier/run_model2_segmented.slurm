#!/bin/bash
#SBATCH --job-name=model2seg
#SBATCH --output=logs/model2seg_%A_%a.out
#SBATCH --error=logs/model2seg_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --signal=B:TERM@120

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Submit ranges:
#   QKeras only: sbatch --array=0-11 run_model2_segmented.slurm
#   LGN only:    sbatch --array=12-35 run_model2_segmented.slurm
#   All:         sbatch --array=0-35 run_model2_segmented.slurm

DATA_DIR="${DATA_DIR:-/scratch/gpfs/IOJALVO/mb7126/SmartPixels/giuData/data/ds8_only/dec6_ds8_quant}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-model2_segmented_top2_${RUN_DATE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/SLURM/${RUN_TAG}}"
RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_ROOT}/results.csv}"

QKERAS_BATCH_SIZE="${QKERAS_BATCH_SIZE:-1024}"
QKERAS_LR="${QKERAS_LR:-0.001}"
QKERAS_EPOCHS="${QKERAS_EPOCHS:-150}"
QKERAS_PATIENCE="${QKERAS_PATIENCE:-20}"

LGN_BATCH_SIZE="${LGN_BATCH_SIZE:-1024}"
LGN_LR="${LGN_LR:-0.01}"
LGN_EPOCHS="${LGN_EPOCHS:-0}"
LGN_EVAL_FREQ="${LGN_EVAL_FREQ:-200}"
LGN_MAX_STEPS="${LGN_MAX_STEPS:-}"
LGN_COMPILE="${LGN_COMPILE:-1}"
LGN_DISABLE_EARLY_STOPPING="${LGN_DISABLE_EARLY_STOPPING:-1}"

BALANCE_CLASSES="${BALANCE_CLASSES:-0}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.2}"
SEED="${SEED:-42}"

QKERAS_MODELS=(
  "qkeras-model-2-w5a10-adc2a"
)

TAUS=(20 40)
SIZE_LABELS=(
  "s05_currentWide_p577M"
)
LGN_MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnFull"
)

N_LOCALS=12
N_QKERAS=${#QKERAS_MODELS[@]}
N_TAUS=${#TAUS[@]}

ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / N_LOCALS))
LOCAL_ID=$((SLURM_ARRAY_TASK_ID % N_LOCALS))

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --local-id "$LOCAL_ID"
  --device cuda
  --validation-split "$VALIDATION_SPLIT"
  --seed "$SEED"
  --results-file "$RESULTS_FILE"
)

if [[ "$BALANCE_CLASSES" == "1" ]]; then
  COMMON_ARGS+=(--balance-classes)
fi

mkdir -p logs "$OUTPUT_ROOT"

if (( ARCH_INDEX < N_QKERAS )); then
  MODEL="${QKERAS_MODELS[$ARCH_INDEX]}"
  OUTPUT_DIR="${OUTPUT_ROOT}/qkeras/${MODEL}/local_${LOCAL_ID}"
  echo "Running QKeras ${MODEL}, local ${LOCAL_ID}"
  python train_model2_segmented.py \
    "${COMMON_ARGS[@]}" \
    --model "$MODEL" \
    --output "${OUTPUT_ROOT}/qkeras" \
    --batch-size "$QKERAS_BATCH_SIZE" \
    --learning-rate "$QKERAS_LR" \
    --epochs "$QKERAS_EPOCHS" \
    --early-stopping-patience "$QKERAS_PATIENCE"
else
  LGN_INDEX=$((ARCH_INDEX - N_QKERAS))
  SIZE_INDEX=$((LGN_INDEX / N_TAUS))
  TAU_INDEX=$((LGN_INDEX % N_TAUS))
  TAU="${TAUS[$TAU_INDEX]}"
  SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
  MODEL="${LGN_MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
  OUTPUT_DIR="${OUTPUT_ROOT}/lgn/${SIZE_LABEL}/tau_${TAU}/${MODEL}/local_${LOCAL_ID}"
  echo "Running LGN ${MODEL}, local ${LOCAL_ID}"
  echo "Size: ${SIZE_LABEL}"
  echo "Tau: ${TAU}"

  LGN_ARGS=(
    --model "$MODEL"
    --output "${OUTPUT_ROOT}/lgn/${SIZE_LABEL}/tau_${TAU}"
    --batch-size "$LGN_BATCH_SIZE"
    --learning-rate "$LGN_LR"
    --epochs "$LGN_EPOCHS"
    --eval-freq "$LGN_EVAL_FREQ"
  )
  if [[ "$LGN_COMPILE" == "1" ]]; then
    LGN_ARGS+=(--compile-model)
  fi
  if [[ "$LGN_DISABLE_EARLY_STOPPING" == "1" ]]; then
    LGN_ARGS+=(--disable-early-stopping)
  fi
  if [[ -n "$LGN_MAX_STEPS" ]]; then
    LGN_ARGS+=(--max-steps "$LGN_MAX_STEPS")
  fi

  python train_model2_segmented.py \
    "${COMMON_ARGS[@]}" \
    "${LGN_ARGS[@]}"
fi

echo "Output root: ${OUTPUT_ROOT}"
echo "Per-run output: ${OUTPUT_DIR}"
echo "Aggregate results: ${RESULTS_FILE}"
