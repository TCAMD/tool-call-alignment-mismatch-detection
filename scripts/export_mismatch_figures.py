# Paste this whole file as one notebook cell after the ratio-study experiment has finished.
# It exports paper-ready figures with "Mismatch" terminology.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


INDIVIDUAL_FIG_DIR = RUN_ROOT / "paper_mismatch_individual_figures"
INDIVIDUAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

RATIO_ORDER_MISMATCH = ["1:1", "7:3", "8:2", "10:1", "100:1"]
MODEL_ORDER_MISMATCH = ["lightgbm", "xgboost", "random_forest", "isolation_forest"]
MODEL_LABELS_MISMATCH = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
}
MODEL_COLORS_MISMATCH = {
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


def save_current_figure(stem):
    png_path = INDIVIDUAL_FIG_DIR / f"{stem}.png"
    svg_path = INDIVIDUAL_FIG_DIR / f"{stem}.svg"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", svg_path)
    return png_path, svg_path


# Load metric results if this cell is run in a fresh kernel after the experiment.
if "ratio_metrics_df" not in globals():
    ratio_metrics_df = pd.read_csv(RUN_ROOT / "ratio_metrics_comparison.csv")

for col in ["precision", "recall", "f1", "accuracy", "auroc", "auprc"]:
    ratio_metrics_df[col] = pd.to_numeric(ratio_metrics_df[col], errors="coerce")

ratio_metrics_df["ratio_label"] = pd.Categorical(
    ratio_metrics_df["ratio_label"],
    categories=RATIO_ORDER_MISMATCH,
    ordered=True,
)

# 1. Save each ratio-study metric curve separately.
metric_specs = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("f1", "F1-score"),
    ("recall", "Recall"),
]
x_positions = np.arange(len(RATIO_ORDER_MISMATCH))

for metric_col, metric_label in metric_specs:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for model_name in MODEL_ORDER_MISMATCH:
        subset = (
            ratio_metrics_df[ratio_metrics_df["model_name"] == model_name]
            .set_index("ratio_label")
            .reindex(RATIO_ORDER_MISMATCH)
        )
        ax.plot(
            x_positions,
            subset[metric_col].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            markersize=5,
            color=MODEL_COLORS_MISMATCH[model_name],
            label=MODEL_LABELS_MISMATCH[model_name],
        )

    ax.set_title(metric_label)
    ax.set_xlabel("Normal:Mismatch Ratio")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(RATIO_ORDER_MISMATCH)
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="best")
    fig.tight_layout()

    save_current_figure(f"mismatch_metric_curve_{metric_col}")
    plt.show()


# 2. Delta embedding L2 norm, PCA, and centroid-projection figures.
if "feature_df" not in globals() or "feature_cols" not in globals():
    raise RuntimeError("feature_df and feature_cols are required. Run the delta feature generation cells first.")

ANALYSIS_SAMPLE_PER_LABEL_LOCAL = globals().get("ANALYSIS_SAMPLE_PER_LABEL", 10000)
SILHOUETTE_SAMPLE_PER_LABEL_LOCAL = globals().get("SILHOUETTE_SAMPLE_PER_LABEL", 1000)

analysis_parts = []
for label_value in [0, 1]:
    subset = feature_df[feature_df["label"] == label_value]
    sample_size = min(ANALYSIS_SAMPLE_PER_LABEL_LOCAL, len(subset))
    analysis_parts.append(subset.sample(n=sample_size, random_state=RANDOM_SEED + label_value))

analysis_df = (
    pd.concat(analysis_parts, ignore_index=True)
    .sample(frac=1, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)

analysis_delta_matrix = analysis_df[feature_cols].to_numpy(dtype=np.float32)
analysis_labels = analysis_df["label"].to_numpy(dtype=int)
analysis_delta_norms = np.linalg.norm(analysis_delta_matrix, axis=1).astype(np.float32)

plot_df = analysis_df[["label"]].copy()
plot_df["delta_l2_norm"] = analysis_delta_norms

# 2-1. L2 norm distribution.
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for label_value, color, label_name in [
    (0, "#1f77b4", "Normal (label 0)"),
    (1, "#d62728", "Mismatch (label 1)"),
]:
    subset = plot_df[plot_df["label"] == label_value]
    ax.hist(
        subset["delta_l2_norm"],
        bins=40,
        alpha=0.55,
        density=True,
        color=color,
        label=label_name,
    )

ax.set_title("Delta Embedding L2 Norm Distribution")
ax.set_xlabel("L2 norm of prompt_emb - tool_emb")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
save_current_figure("mismatch_delta_l2_norm_histogram")
plt.show()

l2_summary_df = (
    plot_df.groupby("label")["delta_l2_norm"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)
l2_quantile_df = (
    plot_df.groupby("label")["delta_l2_norm"]
    .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    .unstack()
    .reset_index()
)
l2_summary_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_l2_norm_summary.csv"
l2_quantile_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_l2_norm_quantiles.csv"
l2_summary_df.to_csv(l2_summary_path, index=False, encoding="utf-8-sig")
l2_quantile_df.to_csv(l2_quantile_path, index=False, encoding="utf-8-sig")
print("Saved:", l2_summary_path)
print("Saved:", l2_quantile_path)
display(l2_summary_df)
display(l2_quantile_df)

# 2-2. PCA scatter.
pca = PCA(n_components=2, random_state=RANDOM_SEED)
delta_pca_2d = pca.fit_transform(analysis_delta_matrix)

pca_df = plot_df.copy()
pca_df["pca_x"] = delta_pca_2d[:, 0]
pca_df["pca_y"] = delta_pca_2d[:, 1]

fig, ax = plt.subplots(figsize=(7.2, 5.4))
for label_value, color, label_name in [
    (0, "#1f77b4", "Normal (label 0)"),
    (1, "#d62728", "Mismatch (label 1)"),
]:
    subset = pca_df[pca_df["label"] == label_value]
    ax.scatter(
        subset["pca_x"],
        subset["pca_y"],
        s=12,
        alpha=0.55,
        color=color,
        label=label_name,
    )

ax.set_title("Delta Embedding PCA")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()
fig.tight_layout()
save_current_figure("mismatch_delta_pca_scatter")
plt.show()

pca_csv_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_pca_sample_coordinates.csv"
pca_summary_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_pca_summary.csv"
pca_df.to_csv(pca_csv_path, index=False, encoding="utf-8-sig")
pca_summary_df = pd.DataFrame(
    {
        "component": ["PCA1", "PCA2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }
)
pca_summary_df.to_csv(pca_summary_path, index=False, encoding="utf-8-sig")
print("Saved:", pca_csv_path)
print("Saved:", pca_summary_path)
display(pca_summary_df)

# 2-3. Projection on centroid-difference axis.
normal_matrix = analysis_delta_matrix[analysis_labels == 0]
mismatch_matrix = analysis_delta_matrix[analysis_labels == 1]

normal_centroid = normal_matrix.mean(axis=0)
mismatch_centroid = mismatch_matrix.mean(axis=0)
centroid_delta = mismatch_centroid - normal_centroid
centroid_delta_norm = float(np.linalg.norm(centroid_delta))

normal_centroid_norm = float(np.linalg.norm(normal_centroid))
mismatch_centroid_norm = float(np.linalg.norm(mismatch_centroid))
centroid_cosine = float(
    np.dot(normal_centroid, mismatch_centroid)
    / max(normal_centroid_norm * mismatch_centroid_norm, 1e-12)
)

normal_spread = np.linalg.norm(normal_matrix - normal_centroid, axis=1)
mismatch_spread = np.linalg.norm(mismatch_matrix - mismatch_centroid, axis=1)

if centroid_delta_norm > 0:
    separation_axis = centroid_delta / centroid_delta_norm
else:
    separation_axis = np.zeros_like(centroid_delta)

projection_scores = analysis_delta_matrix @ separation_axis
projection_df = pd.DataFrame(
    {
        "label": analysis_labels,
        "projection_score": projection_scores,
    }
)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for label_value, color, label_name in [
    (0, "#1f77b4", "Normal (label 0)"),
    (1, "#d62728", "Mismatch (label 1)"),
]:
    subset = projection_df[projection_df["label"] == label_value]
    ax.hist(
        subset["projection_score"],
        bins=40,
        alpha=0.55,
        density=True,
        color=color,
        label=label_name,
    )

ax.set_title("Projection on Centroid-Difference Axis")
ax.set_xlabel("Projection score")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
save_current_figure("mismatch_delta_centroid_projection_histogram")
plt.show()

projection_csv_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_centroid_projection_scores.csv"
projection_quantile_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_centroid_projection_quantiles.csv"
separation_summary_path = INDIVIDUAL_FIG_DIR / "mismatch_delta_separation_summary.csv"
projection_df.to_csv(projection_csv_path, index=False, encoding="utf-8-sig")

projection_quantile_df = (
    projection_df.groupby("label")["projection_score"]
    .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    .unstack()
    .reset_index()
)
projection_quantile_df.to_csv(projection_quantile_path, index=False, encoding="utf-8-sig")

silhouette_parts = []
for label_value in [0, 1]:
    subset = analysis_df[analysis_df["label"] == label_value]
    sample_size = min(SILHOUETTE_SAMPLE_PER_LABEL_LOCAL, len(subset))
    silhouette_parts.append(subset.sample(n=sample_size, random_state=RANDOM_SEED + 100 + label_value))

silhouette_df = pd.concat(silhouette_parts, ignore_index=True)
silhouette_matrix = silhouette_df[feature_cols].to_numpy(dtype=np.float32)
silhouette_labels = silhouette_df["label"].to_numpy(dtype=int)

try:
    silhouette = float(silhouette_score(silhouette_matrix, silhouette_labels, metric="euclidean"))
except Exception as exc:
    silhouette = None
    print("Silhouette score could not be computed:", exc)

separation_summary_df = pd.DataFrame(
    [
        {"metric": "centroid_l2_distance", "value": centroid_delta_norm},
        {"metric": "centroid_cosine_similarity", "value": centroid_cosine},
        {"metric": "normal_within_class_mean_l2", "value": float(normal_spread.mean())},
        {"metric": "mismatch_within_class_mean_l2", "value": float(mismatch_spread.mean())},
        {
            "metric": "normal_projection_mean",
            "value": float(projection_df.loc[projection_df["label"] == 0, "projection_score"].mean()),
        },
        {
            "metric": "mismatch_projection_mean",
            "value": float(projection_df.loc[projection_df["label"] == 1, "projection_score"].mean()),
        },
        {"metric": "silhouette_score_sampled", "value": silhouette},
    ]
)
separation_summary_df.to_csv(separation_summary_path, index=False, encoding="utf-8-sig")
print("Saved:", projection_csv_path)
print("Saved:", projection_quantile_path)
print("Saved:", separation_summary_path)
display(projection_quantile_df)
display(separation_summary_df)

print()
print("All mismatch-labeled outputs saved to:", INDIVIDUAL_FIG_DIR)
