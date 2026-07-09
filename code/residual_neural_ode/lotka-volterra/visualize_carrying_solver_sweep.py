import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIG_ORDER = [
    "euler_ratio1",
    "euler_ratio16",
    "heun_ratio1",
    "rk4_ratio4",
    "diffrax_tsit5",
]
METRICS = [
    "best_loss",
    "validation_mse",
    "extrapolate_mse",
    "final_mse",
    "residual_mse",
    "model_vf_rel_error",
]
PHYSICS_PARAMS = [
    ("a", "learned_f_physics_a", 1.0),
    ("b", "learned_f_physics_b", 0.05),
    ("r", "learned_f_physics_r", 1.5),
    ("z", "learned_f_physics_z", 0.03),
]


def config_sort_key(config_name):
    if config_name in CONFIG_ORDER:
        return CONFIG_ORDER.index(config_name)
    return len(CONFIG_ORDER), config_name


def parse_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    value = str(value).strip()
    if value == "" or value.upper() in {"N/A", "NULL", "NONE", "NAN", "ADAPTIVE"}:
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


def sorted_configs(rows):
    return sorted(
        {row["training_config"] for row in rows if row.get("training_config")},
        key=config_sort_key,
    )


def sorted_seeds(rows):
    seeds = {parse_int(row.get("seed")) for row in rows}
    return sorted(seed for seed in seeds if seed is not None)


def subset(rows, config_name=None, seed=None):
    out = rows
    if config_name is not None:
        out = [row for row in out if row.get("training_config") == config_name]
    if seed is not None:
        out = [row for row in out if parse_int(row.get("seed")) == seed]
    return out


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
            "median": np.nan,
            "range": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size < 2:
        return np.nan
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def write_group_summary(rows, output_dir):
    summary_rows = []
    for config_name in sorted_configs(rows):
        group = subset(rows, config_name=config_name)
        summary = {
            "training_config": config_name,
            "training_method": group[0].get("training_method", ""),
            "ratio": group[0].get("ratio", ""),
            "n_runs": len(group),
            "seeds": ";".join(str(seed) for seed in sorted_seeds(group)),
        }
        for metric in METRICS:
            stats = summarize(metric_values(group, metric))
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summary_rows.append(summary)

    if not summary_rows:
        return
    with (output_dir / "carrying_solver_sweep_group_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_correlation_summary(rows, output_dir):
    pairs = [
        ("best_loss", "extrapolate_mse"),
        ("validation_mse", "extrapolate_mse"),
        ("residual_mse", "extrapolate_mse"),
        ("model_vf_rel_error", "extrapolate_mse"),
    ]
    out_rows = []
    groups = [("all", rows)]
    for config_name in sorted_configs(rows):
        groups.append((config_name, subset(rows, config_name=config_name)))

    for group_name, group_rows in groups:
        for x_metric, y_metric in pairs:
            x = [parse_float(row.get(x_metric)) for row in group_rows]
            y = [parse_float(row.get(y_metric)) for row in group_rows]
            out_rows.append(
                {
                    "group": group_name,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": len(group_rows),
                    "correlation": correlation(x, y),
                }
            )

    with (output_dir / "carrying_solver_sweep_correlations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def write_parameter_recovery_summary(rows, output_dir):
    summary_rows = []
    for config_name in sorted_configs(rows):
        group = subset(rows, config_name=config_name)
        if not group:
            continue
        for param_name, column, true_value in PHYSICS_PARAMS:
            values = metric_values(group, column)
            abs_errors = np.abs(values - true_value)
            rel_errors = abs_errors / (abs(true_value) + 1e-12)
            value_stats = summarize(values)
            abs_stats = summarize(abs_errors)
            rel_stats = summarize(rel_errors)
            summary_rows.append(
                {
                    "training_config": config_name,
                    "parameter": param_name,
                    "true_value": true_value,
                    "learned_mean": value_stats["mean"],
                    "learned_std": value_stats["std"],
                    "learned_min": value_stats["min"],
                    "learned_max": value_stats["max"],
                    "abs_error_mean": abs_stats["mean"],
                    "abs_error_std": abs_stats["std"],
                    "relative_error_mean": rel_stats["mean"],
                    "relative_error_std": rel_stats["std"],
                    "n_runs": len(values),
                }
            )

    if not summary_rows:
        return
    with (output_dir / "carrying_parameter_recovery_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_metric_by_config(rows, output_dir, metric, ylabel):
    configs = sorted_configs(rows)
    x_positions = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(configs)), 5.2))
    for idx, config_name in enumerate(configs):
        values = metric_values(subset(rows, config_name=config_name), metric)
        if values.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else [0.0]
        ax.scatter(x_positions[idx] + np.asarray(jitter), values, alpha=0.75)
        stats = summarize(values)
        ax.errorbar(
            x_positions[idx],
            stats["mean"],
            yerr=stats["std"],
            color="black",
            marker="s",
            capsize=4,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Training Configuration")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"carrying_{metric}_by_config.png", dpi=220)
    plt.close(fig)


def plot_parameter_recovery(rows, output_dir):
    configs = sorted_configs(rows)
    x_positions = np.arange(len(configs))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), squeeze=False)

    for ax, (param_name, column, true_value) in zip(axes.ravel(), PHYSICS_PARAMS):
        for idx, config_name in enumerate(configs):
            values = metric_values(subset(rows, config_name=config_name), column)
            if values.size == 0:
                continue
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else [0.0]
            ax.scatter(
                x_positions[idx] + np.asarray(jitter),
                values,
                alpha=0.75,
                s=28,
            )
            stats = summarize(values)
            ax.errorbar(
                x_positions[idx],
                stats["mean"],
                yerr=stats["std"],
                color="black",
                marker="s",
                capsize=4,
            )
        ax.axhline(
            true_value,
            color="tab:red",
            linestyle="--",
            linewidth=1.3,
            label=f"true {param_name}={true_value:g}",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(configs, rotation=25, ha="right")
        ax.set_ylabel(f"learned {param_name}")
        ax.set_title(f"Parameter Recovery: {param_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Physics Parameter Recovery by Training Configuration", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "carrying_parameter_recovery.png", dpi=220)
    plt.close(fig)


def plot_parameter_recovery_error(rows, output_dir):
    configs = sorted_configs(rows)
    x_positions = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for param_name, column, true_value in PHYSICS_PARAMS:
        means = []
        stds = []
        for config_name in configs:
            values = metric_values(subset(rows, config_name=config_name), column)
            errors = np.abs(values - true_value)
            stats = summarize(errors)
            means.append(stats["mean"])
            stds.append(stats["std"])
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        mask = np.isfinite(means) & (means > 0)
        if np.any(mask):
            ax.errorbar(
                x_positions[mask],
                means[mask],
                yerr=stds[mask],
                marker="o",
                capsize=4,
                label=param_name,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("absolute parameter error")
    ax.set_title("Physics Parameter Recovery Error")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "carrying_parameter_recovery_error.png", dpi=220)
    plt.close(fig)


def plot_seed_trends(rows, output_dir, metric):
    configs = sorted_configs(rows)
    seeds = sorted_seeds(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in seeds:
        values = []
        for config_name in configs:
            group = subset(rows, config_name=config_name, seed=seed)
            values.append(parse_float(group[0].get(metric)) if group else np.nan)
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values) & (values > 0)
        if np.any(mask):
            ax.plot(np.arange(len(configs))[mask], values[mask], marker="o", label=f"seed={seed}")

    means = []
    for config_name in configs:
        values = metric_values(subset(rows, config_name=config_name), metric)
        means.append(np.mean(values) if values.size else np.nan)
    means = np.asarray(means)
    mask = np.isfinite(means) & (means > 0)
    if np.any(mask):
        ax.plot(np.arange(len(configs))[mask], means[mask], color="black", linewidth=2.4, marker="s", label="seed mean")

    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(metric)
    ax.set_title(f"Seed-Level Trends: {metric}")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"carrying_seed_trends_{metric}.png", dpi=220)
    plt.close(fig)


def plot_scatter(rows, output_dir, x_metric, y_metric):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for config_name in sorted_configs(rows):
        group = subset(rows, config_name=config_name)
        x = np.array([parse_float(row.get(x_metric)) for row in group])
        y = np.array([parse_float(row.get(y_metric)) for row in group])
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], label=f"{config_name}, r={correlation(x, y):.2f}", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{x_metric} vs {y_metric}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"carrying_{x_metric}_vs_{y_metric}.png", dpi=220)
    plt.close(fig)


def plot_error_over_time(details, output_dir, t_final=20.0, extrap_start=5.0):
    if not details:
        return
    first = next(iter(details.values()))
    n_times = len(first["extrapolate_mse_by_time"])
    time_grid = np.linspace(0.0, t_final, n_times)
    configs = sorted({key[0] for key in details}, key=config_sort_key)
    fig, axes = plt.subplots(len(configs), 1, figsize=(9, 3.0 * len(configs)), sharex=True, squeeze=False)
    for ax, config_name in zip(axes[:, 0], configs):
        config_curves = [
            curves["extrapolate_mse_by_time"]
            for key, curves in details.items()
            if key[0] == config_name
        ]
        if not config_curves:
            continue
        curve_array = np.asarray(config_curves, dtype=float)
        log_curves = np.log10(np.maximum(curve_array, 1e-16))
        center = 10 ** np.mean(log_curves, axis=0)
        spread = np.std(log_curves, axis=0, ddof=1) if curve_array.shape[0] >= 2 else np.zeros_like(center)
        lower = 10 ** (np.mean(log_curves, axis=0) - spread)
        upper = 10 ** (np.mean(log_curves, axis=0) + spread)
        ax.semilogy(time_grid, center, label="seed log-mean")
        ax.fill_between(time_grid, lower, upper, alpha=0.18, label="+/-1 log-std")
        ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
        ax.set_ylabel("MSE")
        ax.set_title(config_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("t")
    fig.suptitle("Extrapolation Error over Time")
    fig.tight_layout()
    fig.savefig(output_dir / "carrying_extrapolate_mse_over_time.png", dpi=220)
    plt.close(fig)


def visualize(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(input_dir / "carrying_solver_sweep_summary.csv")
    payload = load_pickle(input_dir / "carrying_solver_sweep_details.pkl")
    details = payload.get("details", {})

    write_group_summary(rows, output_dir)
    write_correlation_summary(rows, output_dir)
    write_parameter_recovery_summary(rows, output_dir)
    plot_metric_by_config(rows, output_dir, "extrapolate_mse", "Extrapolation MSE")
    plot_metric_by_config(rows, output_dir, "validation_mse", "Validation MSE")
    plot_metric_by_config(rows, output_dir, "residual_mse", "Residual MSE")
    plot_metric_by_config(rows, output_dir, "best_loss", "Best Validation Loss")
    plot_parameter_recovery(rows, output_dir)
    plot_parameter_recovery_error(rows, output_dir)
    plot_seed_trends(rows, output_dir, "extrapolate_mse")
    plot_seed_trends(rows, output_dir, "residual_mse")
    plot_scatter(rows, output_dir, "best_loss", "extrapolate_mse")
    plot_scatter(rows, output_dir, "residual_mse", "extrapolate_mse")
    plot_error_over_time(details, output_dir)
    print(f"Saved carrying sweep visualizations to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize carrying-capacity solver training sweep."
    )
    parser.add_argument("--input-dir", default="carrying_solver_sweep_results")
    parser.add_argument("--output-dir", default="carrying_solver_sweep_visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
