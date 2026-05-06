# Experiment Reproducibility

## Environment

Install the package versions in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The reported ratio-study notebooks used CPU inference for the downstream models. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

## Rebuild Dataset From Raw JSON

```bash
python scripts/build_toolbench_mismatch_dataset.py
```

Raw ToolBench JSON files should be placed under `data/raw/toolbench/G1_answer/` before rebuilding.

## Rerun Experiments

Run:

1. `notebooks/01_toolbench_mismatch_ratio_study_experiment.ipynb`
2. `notebooks/02_toolbench_mismatch_result_visualization.ipynb`

The visualization notebook can also be run immediately from the copied artifacts in `outputs/runs/paper_ratio_study_results/` without rerunning the full experiment.
