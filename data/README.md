# Data

This repository uses:

- Synthetic datasets generated in `src/causal_inference_lab/data_generators.py` (committed in code).
- The prepared Lalonde / NSW benchmark for notebook `10_lalonde_job_training.ipynb`.

## Lalonde benchmark preparation

`scripts/prepare_lalonde_job_training_dataset.py` downloads the public dataset from:

- `https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv`

It normalizes the columns to the repository's canonical analysis names and writes:

- `data/processed/lalonde_job_training.csv`
- `data/processed/lalonde_job_training_manifest.json`

The preparation script runs automatically from notebook 10 when the processed file is missing.

If you use this benchmark in other work, check the source dataset license and terms in the upstream
`MatchIt` / Lalonde sources before redistributing raw files.
