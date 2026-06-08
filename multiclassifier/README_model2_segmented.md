# Model 2 Segmented Training

This adds a standalone path for apples-to-apples Model 2 training on the
`TrainSetLocal*` / `TestSetLocal*` CSV segmentation used by `train.ipynb`.
It does not use the top-level SmartPixels YAML config or dataloaders.

Default data directory:

```bash
/scratch/gpfs/IOJALVO/mb7126/SmartPixels/giuData/data/ds8_only/dec6_ds8_quant
```

List supported models:

```bash
python train_model2_segmented.py --list-models
```

Run one QKeras Model 2 job for local bin 6:

```bash
python train_model2_segmented.py \
  --model qkeras-model-2-w4a8-adc2a \
  --local-id 6 \
  --device cuda \
  --batch-size 1024 \
  --epochs 150 \
  --output results/model2_segmented/qkeras
```

Submit the two highest-precision QKeras quantization variants across all twelve local bins:

```bash
sbatch --array=0-23 run_model2_segmented.slurm
```

Submit the two largest LGN size variants with tau 20 and 40 across all twelve local bins:

```bash
sbatch --array=24-71 run_model2_segmented.slurm
```

Submit the full reduced comparison:

```bash
sbatch --array=0-71 run_model2_segmented.slurm
```

This is 72 jobs total:

```text
(2 QKeras + 2 LGN sizes * 2 taus) * 12 local bins
```

Useful Slurm overrides:

```bash
BALANCE_CLASSES=1 sbatch --array=0-23 run_model2_segmented.slurm
LGN_MAX_STEPS=5000 sbatch --array=24-71 run_model2_segmented.slurm
```

After the Slurm jobs finish, make the acceptance curves and balanced-accuracy
summary from the saved prediction CSVs:

```bash
python evaluate_acceptance.py \
  --results-root results/SLURM/2026_06_07_model2_segmented_top2 \
  --outdir results/acceptance_model2_segmented
```

The script prints `balanced_accuracy` for each model group and writes
`balanced_accuracy.csv`, `acceptance_bins.csv`, `acceptance.png`, and
`acceptance.pdf`. Use `--group-by run` if you want one curve per local segment
instead of aggregating local segments by model name.

The script preserves the notebook's default preprocessing: no scaling, sparse
integer labels, and padded feature columns `14`, `15`, and `16`. Set
`--no-pad-like-notebook` only if you intentionally want the raw CSV columns.
