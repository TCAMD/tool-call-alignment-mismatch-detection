# Paste this whole file as one notebook cell.
# It exports paper-ready ROC and Precision-Recall curves for the 1:1 Normal:Mismatch experiment.
# It uses saved test_predictions.csv files, so it does not retrain any model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROC_PR_OUT_DIR = RUN_ROOT / "paper_mismatch_individual_figures" / "roc_pr_1_1"
ROC_PR_OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIO_NAME_FOR_CURVES = "1_1"
RATIO_LABEL_FOR_CURVES = "1:1"

MODEL_ORDER_FOR_CURVES = ["lightgbm", "xgboost", "random_forest", "isolation_forest"]
MODEL_LABELS_FOR_CURVES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
}
MODEL_COLORS_FOR_CURVES = {
    "lightgbm": "#1f77b4",
    "xgboost": "#ff7f0e",
    "random_forest": "#2ca02c",
    "isolation_forest": "#d62728",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_roc_pr_fig(fig, stem):
    png_path = ROC_PR_OUT_DIR / f"{stem}.png"
    svg_path = ROC_PR_OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", svg_path)
    return png_path, svg_path


prediction_frames = {}
metric_rows = []

for model_name in MODEL_ORDER_FOR_CURVES:
    pred_path = RUN_ROOT / f"ratio_{RATIO_NAME_FOR_CURVES}" / model_name / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required_cols = {"label", "score"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"{pred_path} is missing columns: {missing_cols}")

    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["score"] = pred_df["score"].astype(float)
    prediction_frames[model_name] = pred_df

    y_true = pred_df["label"].to_numpy(dtype=int)
    y_score = pred_df["score"].to_numpy(dtype=float)

    metric_rows.append(
        {
            "model_name": model_name,
            "model_label": MODEL_LABELS_FOR_CURVES[model_name],
            "ratio_label": RATIO_LABEL_FOR_CURVES,
            "test_rows": int(len(pred_df)),
            "normal_count": int((y_true == 0).sum()),
            "mismatch_count": int((y_true == 1).sum()),
            "mismatch_rate": float(y_true.mean()),
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }
    )

roc_pr_summary_df = pd.DataFrame(metric_rows)
summary_path = ROC_PR_OUT_DIR / "roc_pr_1_1_summary.csv"
roc_pr_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
display(roc_pr_summary_df.round(4))
print("Saved:", summary_path)

mismatch_rate = float(next(iter(prediction_frames.values()))["label"].mean())

# 1. Combined ROC curve for all models.
fig, ax = plt.subplots(figsize=(6.6, 5.4))

for model_name in MODEL_ORDER_FOR_CURVES:
    pred_df = prediction_frames[model_name]
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_score = pred_df["score"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc_value = roc_auc_score(y_true, y_score)
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        color=MODEL_COLORS_FOR_CURVES[model_name],
        label=f"{MODEL_LABELS_FOR_CURVES[model_name]} (AUROC={auroc_value:.4f})",
    )

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
ax.set_title("ROC Curves on the 1:1 Dataset")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_roc_pr_fig(fig, "mismatch_1_1_roc_curves_all_models")
plt.show()

# 2. Combined Precision-Recall curve for all models.
fig, ax = plt.subplots(figsize=(6.6, 5.4))

for model_name in MODEL_ORDER_FOR_CURVES:
    pred_df = prediction_frames[model_name]
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_score = pred_df["score"].to_numpy(dtype=float)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    auprc_value = average_precision_score(y_true, y_score)
    ax.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        color=MODEL_COLORS_FOR_CURVES[model_name],
        label=f"{MODEL_LABELS_FOR_CURVES[model_name]} (AUPRC={auprc_value:.4f})",
    )

ax.axhline(
    mismatch_rate,
    linestyle="--",
    color="gray",
    linewidth=1,
    label=f"Mismatch-rate baseline ({mismatch_rate:.4f})",
)
ax.set_title("Precision-Recall Curves on the 1:1 Dataset")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save_roc_pr_fig(fig, "mismatch_1_1_pr_curves_all_models")
plt.show()

# 3. Per-model ROC/PR panels, one file per model.
for model_name in MODEL_ORDER_FOR_CURVES:
    pred_df = prediction_frames[model_name]
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_score = pred_df["score"].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    auroc_value = roc_auc_score(y_true, y_score)
    auprc_value = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

    axes[0].plot(
        fpr,
        tpr,
        linewidth=2.2,
        color=MODEL_COLORS_FOR_CURVES[model_name],
        label=f"AUROC={auroc_value:.4f}",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall_curve,
        precision_curve,
        linewidth=2.2,
        color=MODEL_COLORS_FOR_CURVES[model_name],
        label=f"AUPRC={auprc_value:.4f}",
    )
    axes[1].axhline(
        mismatch_rate,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Baseline={mismatch_rate:.4f}",
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower left")

    fig.suptitle(
        f"{MODEL_LABELS_FOR_CURVES[model_name]} on the 1:1 Normal:Mismatch Dataset",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_roc_pr_fig(fig, f"mismatch_1_1_roc_pr_{model_name}")
    plt.show()

print()
print("All 1:1 ROC/PR curve outputs saved to:", ROC_PR_OUT_DIR)
