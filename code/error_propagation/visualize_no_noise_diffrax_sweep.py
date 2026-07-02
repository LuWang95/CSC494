import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["heun", "rk4"]
METRICS = [
    "extrapolate_mse",
    "final_mse",
    "validation_mse",
    "best_loss",
    "residual_mse",
    "model_vf_rel_error",
]


def method_sort_key(method):
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER), method


def parse_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    value = str(value).strip()
    if value == "" or value.upper() in {"N/A", "NULL", "NONE", "NAN"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=None):
    number = parse_float(value)
    if not np.isfinite(number):
        return default
    return int(number)


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_pickle(path):
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def metric_values(rows, metric):
    values = np.array([parse_float(row.get(metric)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "median": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "median": float(np.median(values)),
    }


def sorted_methods(rows):
    return sorted(
        {row["training_method"] for row in rows if row.get("training_method")},
        key=method_sort_key,
    )


def sorted_ratios(rows):
    ratios = {parse_int(row.get("ratio")) for row in rows}
    return sorted(ratio for ratio in ratios if ratio is not None)


def sorted_seeds(rows):
    seeds = {parse_int(row.get("seed")) for row in rows}
    return sorted(seed for seed in seeds if seed is not None)


def subset_rows(rows, method=None, ratio=None, seed=None):
    out = rows
    if method is not None:
        out = [row for row in out if row.get("training_method") == method]
    if ratio is not None:
        out = [row for row in out if parse_int(row.get("ratio")) == ratio]
    if seed is not None:
        out = [row for row in out if parse_int(row.get("seed")) == seed]
    return out


def single_metric(rows, method, ratio, seed, metric):
    matching = subset_rows(rows, method=method, ratio=ratio, seed=seed)
    if not matching:
        return np.nan
    return parse_float(matching[0].get(metric))


def ratio_metric_matrix(rows, method, metric):
    seeds = sorted_seeds(rows)
    ratios = sorted_ratios(rows)
    matrix = np.full((len(seeds), len(ratios)), np.nan)
    for i, seed in enumerate(seeds):
        for j, ratio in enumerate(ratios):
            matrix[i, j] = single_metric(rows, method, ratio, seed, metric)
    return seeds, ratios, matrix


def rank_values(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(values.shape, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan
    return correlation(rank_values(x), rank_values(y))


def write_group_summary(rows, output_dir):
    summary_rows = []
    for method in sorted_methods(rows):
        for ratio in sorted_ratios(rows):
            group = subset_rows(rows, method=method, ratio=ratio)
            if not group:
                continue
            summary = {
                "training_method": method,
                "ratio": ratio,
                "n_seeds": len(group),
            }
            for metric in METRICS:
                stats = summarize(metric_values(group, metric))
                for stat_name, value in stats.items():
                    summary[f"{metric}_{stat_name}"] = value
            summary_rows.append(summary)

    if not summary_rows:
        return
    path = output_dir / "ratio_group_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_seed_ranking_summary(rows, output_dir, metric):
    ranking_rows = []
    consistency_rows = []

    for method in sorted_methods(rows):
        seeds, ratios, matrix = ratio_metric_matrix(rows, method, metric)
        if matrix.size == 0:
            continue

        for i, seed in enumerate(seeds):
            values = matrix[i]
            if not np.any(np.isfinite(values)):
                continue
            best_idx = int(np.nanargmin(values))
            ranked_indices = [
                idx for idx in np.argsort(values) if np.isfinite(values[idx])
            ]
            ranking_rows.append(
                {
                    "training_method": method,
                    "seed": seed,
                    "metric": metric,
                    "best_ratio": ratios[best_idx],
                    "best_value": values[best_idx],
                    "ratio_ranking_low_to_high_error": ";".join(
                        str(ratios[idx]) for idx in ranked_indices
                    ),
                    "values_by_ratio": ";".join(
                        f"{ratio}:{values[j]:.12g}"
                        for j, ratio in enumerate(ratios)
                        if np.isfinite(values[j])
                    ),
                }
            )

        correlations = []
        for i, seed_i in enumerate(seeds):
            for j, seed_j in enumerate(seeds):
                if j <= i:
                    continue
                corr = spearman(matrix[i], matrix[j])
                if np.isfinite(corr):
                    correlations.append(corr)
                    consistency_rows.append(
                        {
                            "training_method": method,
                            "metric": metric,
                            "seed_i": seed_i,
                            "seed_j": seed_j,
                            "spearman_rank_correlation": corr,
                        }
                    )

        if correlations:
            consistency_rows.append(
                {
                    "training_method": method,
                    "metric": metric,
                    "seed_i": "mean_pairwise",
                    "seed_j": "mean_pairwise",
                    "spearman_rank_correlation": float(np.mean(correlations)),
                }
            )

    if ranking_rows:
        with (output_dir / f"seed_ratio_rankings_{metric}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ranking_rows)

    if consistency_rows:
        with (output_dir / f"seed_rank_consistency_{metric}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(consistency_rows[0].keys()))
            writer.writeheader()
            writer.writerows(consistency_rows)


def plot_ratio_mean_std(rows, output_dir):
    metrics = [
        ("extrapolate_mse", "Extrapolation MSE"),
        ("final_mse", "Final MSE"),
        ("best_loss", "Best Training Loss"),
        ("residual_mse", "Residual MSE"),
    ]
    methods = sorted_methods(rows)
    ratios = sorted_ratios(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for method in methods:
            means = []
            stds = []
            valid_ratios = []
            for ratio in ratios:
                values = metric_values(subset_rows(rows, method=method, ratio=ratio), metric)
                if values.size == 0:
                    continue
                stats = summarize(values)
                means.append(stats["mean"])
                stds.append(stats["std"])
                valid_ratios.append(ratio)
            means = np.asarray(means, dtype=float)
            stds = np.asarray(stds, dtype=float)
            valid_ratios = np.asarray(valid_ratios, dtype=float)
            mask = np.isfinite(means) & (means > 0.0)
            if np.any(mask):
                ax.errorbar(
                    valid_ratios[mask],
                    means[mask],
                    yerr=stds[mask],
                    marker="o",
                    capsize=4,
                    label=method,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ratios)
        ax.set_xticklabels(ratios)
        ax.set_xlabel("train ratio")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: mean +/- seed std")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Does Larger Training Ratio Improve Average Performance?", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_mean_std_key_metrics.png", dpi=220)
    plt.close(fig)


def plot_seed_trends(rows, output_dir, metric):
    methods = sorted_methods(rows)
    ratios = sorted_ratios(rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), squeeze=False)

    for ax, method in zip(axes[0], methods):
        seeds, _, matrix = ratio_metric_matrix(rows, method, metric)
        for i, seed in enumerate(seeds):
            values = matrix[i]
            mask = np.isfinite(values) & (values > 0.0)
            if np.any(mask):
                ax.plot(
                    np.asarray(ratios)[mask],
                    values[mask],
                    marker="o",
                    alpha=0.8,
                    label=f"seed={seed}",
                )

        group_means = []
        for ratio in ratios:
            values = metric_values(subset_rows(rows, method=method, ratio=ratio), metric)
            group_means.append(np.mean(values) if values.size else np.nan)
        group_means = np.asarray(group_means, dtype=float)
        mask = np.isfinite(group_means) & (group_means > 0.0)
        if np.any(mask):
            ax.plot(
                np.asarray(ratios)[mask],
                group_means[mask],
                color="black",
                linewidth=2.5,
                marker="s",
                label="seed mean",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ratios)
        ax.set_xticklabels(ratios)
        ax.set_xlabel("train ratio")
        ax.set_ylabel(metric)
        ax.set_title(method)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Seed-Level Trends Across Train Ratios: {metric}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"seed_trends_by_ratio_{metric}.png", dpi=220)
    plt.close(fig)


def plot_seed_ratio_heatmaps(rows, output_dir, metric):
    methods = sorted_methods(rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.8), squeeze=False)
    values_all = []
    matrices = {}

    for method in methods:
        seeds, ratios, matrix = ratio_metric_matrix(rows, method, metric)
        matrices[method] = (seeds, ratios, matrix)
        values_all.extend(matrix[np.isfinite(matrix) & (matrix > 0.0)])

    values_all = np.asarray(values_all, dtype=float)
    if values_all.size == 0:
        return
    vmin = np.log10(values_all).min()
    vmax = np.log10(values_all).max()

    for ax, method in zip(axes[0], methods):
        seeds, ratios, matrix = matrices[method]
        image = np.full_like(matrix, np.nan, dtype=float)
        mask = np.isfinite(matrix) & (matrix > 0.0)
        image[mask] = np.log10(matrix[mask])
        im = ax.imshow(image, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_xticklabels(ratios)
        ax.set_yticks(np.arange(len(seeds)))
        ax.set_yticklabels(seeds)
        ax.set_xlabel("train ratio")
        ax.set_ylabel("seed")
        ax.set_title(method)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.1e}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )

    fig.suptitle(f"Seed x Ratio Heatmap: {metric}", fontsize=14)
    fig.subplots_adjust(right=0.9, top=0.85, wspace=0.35)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    fig.colorbar(im, cax=cbar_ax, label=f"log10({metric})")
    fig.savefig(output_dir / f"seed_ratio_heatmap_{metric}.png", dpi=220)
    plt.close(fig)


def plot_optimization_generalization(rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), squeeze=False)
    x_metrics = [
        ("best_loss", "best training loss"),
        ("validation_mse", "validation MSE"),
    ]

    for ax, (x_metric, x_label) in zip(axes[0], x_metrics):
        for method in sorted_methods(rows):
            method_rows = subset_rows(rows, method=method)
            x = np.array([parse_float(row.get(x_metric)) for row in method_rows])
            y = np.array([parse_float(row.get("extrapolate_mse")) for row in method_rows])
            ratios = np.array([parse_int(row.get("ratio")) for row in method_rows])
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if not np.any(mask):
                continue
            scatter = ax.scatter(
                x[mask],
                y[mask],
                c=ratios[mask],
                cmap="viridis",
                alpha=0.8,
                label=f"{method}, r={correlation(x, y):.2f}",
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("extrapolation MSE")
        ax.set_title(f"Does {x_label} explain extrapolation?")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.subplots_adjust(right=0.9, wspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(scatter, cax=cbar_ax, label="train ratio")
    fig.savefig(output_dir / "optimization_vs_generalization_scatter.png", dpi=220)
    plt.close(fig)


def plot_generalization_gap(rows, output_dir):
    methods = sorted_methods(rows)
    ratios = sorted_ratios(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for method in methods:
        means = []
        stds = []
        valid_ratios = []
        for ratio in ratios:
            group = subset_rows(rows, method=method, ratio=ratio)
            gaps = []
            for row in group:
                extrap = parse_float(row.get("extrapolate_mse"))
                best = parse_float(row.get("best_loss"))
                if np.isfinite(extrap) and np.isfinite(best) and best > 0.0:
                    gaps.append(extrap / best)
            if not gaps:
                continue
            stats = summarize(np.asarray(gaps, dtype=float))
            means.append(stats["mean"])
            stds.append(stats["std"])
            valid_ratios.append(ratio)
        means = np.asarray(means)
        stds = np.asarray(stds)
        valid_ratios = np.asarray(valid_ratios)
        mask = np.isfinite(means) & (means > 0.0)
        if np.any(mask):
            ax.errorbar(
                valid_ratios[mask],
                means[mask],
                yerr=stds[mask],
                marker="o",
                capsize=4,
                label=method,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels(ratios)
    ax.set_xlabel("train ratio")
    ax.set_ylabel("extrapolate_mse / best_loss")
    ax.set_title("Generalization Gap Proxy Across Train Ratios")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "generalization_gap_by_ratio.png", dpi=220)
    plt.close(fig)


def plot_error_curves(details, output_dir, t_final, extrap_start, max_curves):
    if not details:
        return
    first = next(iter(details.values()))
    curve = np.asarray(first.get("extrapolate_mse_by_time", []), dtype=float)
    if curve.size == 0:
        return
    time_grid = np.linspace(0.0, t_final, curve.size)

    ranked = []
    for key, curves in details.items():
        mse = np.asarray(curves.get("extrapolate_mse_by_time", []), dtype=float)
        if mse.size:
            ranked.append((float(mse[-1]), key, mse))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:max_curves]

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, key, mse in selected:
        seed, method, ratio, _, _ = key
        ax.semilogy(time_grid, mse, label=f"{method}, ratio={ratio}, seed={seed}")
    ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE by time")
    ax.set_title(f"Top {len(selected)} Runs by Final Error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "top_runs_error_over_time.png", dpi=220)
    plt.close(fig)


def visualize(args):
    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(input_dir / "prediction_sweep_summary.csv")
    payload = load_pickle(input_dir / "prediction_sweep_details.pkl")
    details = payload.get("details", {})
    if not rows:
        rows = payload.get("rows", [])

    write_group_summary(rows, output_dir)
    write_seed_ranking_summary(rows, output_dir, "extrapolate_mse")
    write_seed_ranking_summary(rows, output_dir, "final_mse")
    plot_ratio_mean_std(rows, output_dir)
    plot_seed_trends(rows, output_dir, "extrapolate_mse")
    plot_seed_trends(rows, output_dir, "best_loss")
    plot_seed_ratio_heatmaps(rows, output_dir, "extrapolate_mse")
    plot_seed_ratio_heatmaps(rows, output_dir, "best_loss")
    plot_optimization_generalization(rows, output_dir)
    plot_generalization_gap(rows, output_dir)
    plot_error_curves(
        details,
        output_dir,
        t_final=args.t_final,
        extrap_start=args.extrap_start,
        max_curves=args.max_curves,
    )

    print(f"Saved no-noise Diffrax sweep visualizations to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize no-noise multi-seed ratio sweep with Diffrax evaluation."
    )
    parser.add_argument(
        "--input-dir",
        default="no_noise_diffrax_prediction_sweep_results",
        help="Directory containing prediction_sweep_summary.csv and details pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default="no_noise_diffrax_visualizations",
        help="Directory for plots and analysis CSVs.",
    )
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--extrap-start", type=float, default=5.0)
    parser.add_argument("--max-curves", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
