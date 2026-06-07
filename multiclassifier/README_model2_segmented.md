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

Submit all five QKeras quantization variants across all twelve local bins:

```bash
sbatch --array=0-59 run_model2_segmented.slurm
```

Submit the 25 LGN size/tau variants across all twelve local bins:

```bash
sbatch --array=60-359 run_model2_segmented.slurm
```

Useful Slurm overrides:

```bash
BALANCE_CLASSES=1 sbatch --array=0-59 run_model2_segmented.slurm
LGN_MAX_STEPS=5000 sbatch --array=60-359 run_model2_segmented.slurm
```

The script preserves the notebook's default preprocessing: no scaling, sparse
integer labels, and padded feature columns `14`, `15`, and `16`. Set
`--no-pad-like-notebook` only if you intentionally want the raw CSV columns.
