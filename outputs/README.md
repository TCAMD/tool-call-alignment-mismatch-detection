# Output Artifacts

The copied paper run artifacts are distributed through the GitHub Release asset `paper-data-v1.0.zip`.

Recommended extraction command from the repository root:

```powershell
Expand-Archive .\paper-data-v1.0.zip -DestinationPath . -Force
```

The visualization notebook will then find:

- `outputs/runs/paper_ratio_study_results/`

If Windows Explorer creates an extra `paper-data-v1.0` folder during extraction, the notebook can also detect `paper-data-v1.0/outputs/runs/paper_ratio_study_results/`.

This folder contains the saved ratio-study CSVs, paper figures, and per-model test predictions used to regenerate ROC/PR and metric figures without rerunning the full experiment.
