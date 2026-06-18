#!/bin/bash
#SBATCH --job-name=model2seg
#SBATCH --chdir=/home/mb7126/SmartPixels/directional-pixel-detectors/multiclassifier
#SBATCH --output=logs/model2seg_%A_%a.out
#SBATCH --error=logs/model2seg_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --signal=TERM@300

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/home/mb7126/SmartPixels/directional-pixel-detectors/multiclassifier}"
cd "$SCRIPT_DIR"

# Submit:
#   Compact LGN size/tau/local sweep:
#   sbatch run_model2_segmented.slurm

DATA_DIR="${DATA_DIR:-/scratch/gpfs/IOJALVO/mb7126/SmartPixels/giuData/data/ds8_only/dec6_ds8_quant}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-model2_segmented_lgn_sweep_${RUN_DATE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/SLURM/${RUN_TAG}}"
RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_ROOT}/results.csv}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"

LGN_BATCH_SIZE="${LGN_BATCH_SIZE:-1024}"
LGN_LR="${LGN_LR:-0.01}"
LGN_EPOCHS="${LGN_EPOCHS:-0}"
LGN_EVAL_FREQ="${LGN_EVAL_FREQ:-200}"
LGN_COMPILE="${LGN_COMPILE:-1}"

BALANCE_CLASSES="${BALANCE_CLASSES:-0}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.2}"
SEED="${SEED:-42}"

LOCAL_IDS=(0 2 4 7 9 11)
TAUS=(20 40)
SIZE_LABELS=(
  "s_debug_p10M"
  "s_medium_p15M"
  "s_wide_p20M"
)
LGN_MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnDebug_p10M"
  "lgn-dense-2-dense-100_128_-model2lgnMedium_p15M"
  "lgn-dense-2-dense-100_128_-model2lgnWide_p20M"
)

N_LOCALS=${#LOCAL_IDS[@]}
N_TAUS=${#TAUS[@]}
N_LGN_SIZES=${#SIZE_LABELS[@]}
N_LGN_ARCHES=$((N_LGN_SIZES * N_TAUS))
N_JOBS=$((N_LGN_ARCHES * N_LOCALS))

if (( ${#SIZE_LABELS[@]} != ${#LGN_MODEL_PREFIXES[@]} )); then
  echo "SIZE_LABELS and LGN_MODEL_PREFIXES must have the same length." >&2
  exit 2
fi

ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / N_LOCALS))
LOCAL_INDEX=$((SLURM_ARRAY_TASK_ID % N_LOCALS))
LOCAL_ID="${LOCAL_IDS[$LOCAL_INDEX]}"

if (( SLURM_ARRAY_TASK_ID >= N_JOBS )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside configured LGN job count ${N_JOBS}." >&2
  exit 2
fi

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --local-id "$LOCAL_ID"
  --device "$DEVICE"
  --validation-split "$VALIDATION_SPLIT"
  --seed "$SEED"
  --results-file "$RESULTS_FILE"
)

if [[ "$BALANCE_CLASSES" == "1" ]]; then
  COMMON_ARGS+=(--balance-classes)
fi
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

mkdir -p logs "$OUTPUT_ROOT"

SIZE_INDEX=$((ARCH_INDEX / N_TAUS))
TAU_INDEX=$((ARCH_INDEX % N_TAUS))
TAU="${TAUS[$TAU_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${LGN_MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
OUTPUT_DIR="${OUTPUT_ROOT}/lgn/${SIZE_LABEL}/tau_${TAU}/${MODEL}/local_${LOCAL_ID}"

echo "Running LGN ${MODEL}, local ${LOCAL_ID}"
echo "Size: ${SIZE_LABEL}"
echo "Tau: ${TAU}"
echo "Configured jobs: ${N_JOBS} = ${N_LGN_SIZES} sizes * ${N_TAUS} taus * ${N_LOCALS} locals"
echo "LGN stop policy: 5h Slurm walltime only; no early stopping and no max-step cap."

LGN_ARGS=(
  --model "$MODEL"
  --output "${OUTPUT_ROOT}/lgn/${SIZE_LABEL}/tau_${TAU}"
  --batch-size "$LGN_BATCH_SIZE"
  --learning-rate "$LGN_LR"
  --epochs "$LGN_EPOCHS"
  --eval-freq "$LGN_EVAL_FREQ"
  --disable-early-stopping
)
if [[ "$LGN_COMPILE" == "1" ]]; then
  LGN_ARGS+=(--compile-model)
fi

python train_model2_segmented.py \
  "${COMMON_ARGS[@]}" \
  "${LGN_ARGS[@]}"

echo "Output root: ${OUTPUT_ROOT}"
echo "Per-run output: ${OUTPUT_DIR}"
echo "Aggregate results: ${RESULTS_FILE}"
