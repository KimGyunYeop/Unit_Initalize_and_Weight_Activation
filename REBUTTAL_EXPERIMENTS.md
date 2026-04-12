# Rebuttal Experiments

This repository now supports a focused rebuttal suite for middle-layer insertion experiments without changing the legacy CLI.

## Added insertion options

- `--init_type`: `unit`, `xavier`, `he`, `zero`
- `--activation_kind`: `none`, `gelu`, `relu`
- `--activation_placement`: `none`, `weight`, `output`

Legacy `--act_type` remains supported:

- `midgelu` -> `weight + gelu`
- `midrelu` -> `weight + relu`
- `gelu` -> `output + gelu`
- `relu` -> `output + relu`

## Named presets

- `baseline`
- `naive_xavier_output_gelu`
- `naive_he_output_relu`
- `xavier_no_activation`
- `zero_no_activation`
- `unit_no_activation`
- `xavier_weight_gelu`
- `unit_weight_gelu`
- `unit_weight_relu`

`baseline` is implemented as the no-added-layer reference.

## Multi-seed

Each training entry script still supports `--seed`, and also accepts `--seeds` as either:

- comma-separated: `--seeds 13,21,42`
- repeated: `--seeds 13 --seeds 21 --seeds 42`

When multiple seeds are passed, the entry script dispatches child runs one seed at a time and writes one `metrics.json` per completed seed.

## Metrics output

Each completed run writes a machine-readable `metrics.json` under `--run_output_root`, including:

- model
- task
- seed
- preset
- init/activation settings
- insertion position metadata
- final metrics

For DeBERTa GLUE runs, `--diagnostic_mode relative_perturbation` adds an initialization-time diagnostic on inserted linear modules.

## Focused runner scripts

- `scripts/run_rebuttal_glue.sh`
- `scripts/run_rebuttal_generation.sh`
- `scripts/run_rebuttal_vit.sh`
- `scripts/run_rebuttal_all.sh`

These scripts are restart-safe: if a matching `metrics.json` already exists for the same model/task/preset/seed under the selected output root, that run is skipped.

## Aggregation

Use:

```bash
python aggregate_rebuttal_results.py \
  --input_root rebuttal_runs \
  --output_prefix aggregated_rebuttal_results
```

Outputs:

- `aggregated_rebuttal_results.csv`
- `aggregated_rebuttal_results.json`
- `aggregated_rebuttal_results.md`

The aggregation computes:

- seed-wise mean and std
- delta vs the `baseline` preset
- retention ratio vs the no-added-layer baseline
