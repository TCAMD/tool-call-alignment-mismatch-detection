# Dataset Files

Large dataset CSV files are distributed through the GitHub Release asset `paper-data-v1.0.zip`.

Recommended extraction command from the repository root:

```powershell
Expand-Archive .\paper-data-v1.0.zip -DestinationPath . -Force
```

The following paths will then be created:

- `data/datasets/toolbench_mismatch_dataset.csv`
- `data/datasets/ratio_samples/*.csv`

If Windows Explorer creates an extra `paper-data-v1.0` folder during extraction, the notebooks can also detect `paper-data-v1.0/data/datasets/`.
