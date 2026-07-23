import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solver_error_no_noise import T, make_reference_data, t_extrapolate


DEFAULT_SOURCES = [
    (
        "euler_train_diffrax_eval",
        "diffrax_eval",
        "no_noise_euler_diffrax_prediction_sweep_results/prediction_sweep_summary.csv",
    ),
    (
        "heun_rk4_train_diffrax_eval",
        "diffrax_eval",
        "no_noise_diffrax_prediction_sweep_results/prediction_sweep_summary.csv",
    ),
    (
        "diffrax_train_rk4_eval",
        "rk4_high_precision_eval",
        "diffrax_train_rk4_predict_results/prediction_sweep_summary.csv",
    ),
]

DEFAULT_DETAIL_SOURCES = [
    "no_noise_euler_diffrax_prediction_sweep_results/prediction_sweep_details.pkl",
    "no_noise_diffrax_prediction_sweep_results/prediction_sweep_details.pkl",
    "diffrax_train_rk4_predict_results/prediction_sweep_details.pkl",
]

METHOD_ORDER = ["forward_euler", "heun", "rk4", "diffrax_tsit5"]
STATE_SCALE = np.array([50.0, 50.0])
A, B, R, Z, C = 1.0, 0.05, 1.5, 0.03, 0.005
CORE_COLUMNS = [
    "experiment_source",
    "evaluation_protocol",
    "experiment_kind",
    "training_method",
    "prediction_method",
    "ratio",
    "predict_ratio",
    "seed",
    "h_model",
    "h_predict",
    "noise_level",
    "best_loss",
    "train_mse",
    "validation_mse",
    "extrapolate_mse",
    "residual_mse",
    "residual_rel_error",
    "model_vf_mse",
    "model_vf_rel_error",
    "instability_rate",
    "final_mse",
    "final_l2_error",
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


def load_sources(script_dir, sources):
    rows = []
    for source_name, evaluation_protocol, relative_path in sources:
        path = Path(relative_path)
        if not path.is_absolute():
            path = script_dir / path
        source_rows = read_csv_rows(path)
        for row in source_rows:
            merged = {column: row.get(column, "") for column in CORE_COLUMNS}
            merged["experiment_source"] = source_name
            merged["evaluation_protocol"] = evaluation_protocol
            rows.append(merged)
    return rows


def load_trained_models(script_dir, detail_sources):
    trained = {}
    for relative_path in detail_sources:
        path = Path(relative_path)
        if not path.is_absolute():
            path = script_dir / path
        payload = load_pickle(path)
        for key, model in payload.get("trained", {}).items():
            if len(key) != 3:
                continue
            seed, method, ratio = key
            trained[(method, str(ratio), int(seed))] = model
    return trained


def swish(x):
    return x / (1.0 + np.exp(-x))


def nn_numpy(states, nn_params):
    activations = np.asarray(states, dtype=float) / STATE_SCALE
    for weights, bias in nn_params[:-1]:
        activations = swish(activations @ np.asarray(weights).T + np.asarray(bias))
    final_weights, final_bias = nn_params[-1]
    return activations @ np.asarray(final_weights).T + np.asarray(final_bias)


def f_physics_numpy(states, f_physics_params):
    states = np.asarray(states, dtype=float)
    prey = states[:, 0]
    predator = states[:, 1]
    params = np.asarray(f_physics_params, dtype=float)
    return np.stack(
        [
            params[0] * prey - params[1] * predator * prey,
            -params[2] * predator + params[3] * predator * prey,
        ],
        axis=1,
    )


def true_rhs_numpy(states):
    states = np.asarray(states, dtype=float)
    prey = states[:, 0]
    predator = states[:, 1]
    return np.stack(
        [
            A * prey - B * prey * predator - C * prey * prey,
            -R * predator + Z * prey * predator,
        ],
        axis=1,
    )


def model_rhs_numpy(states, params):
    return (
        f_physics_numpy(states, params["f_physics"])
        + nn_numpy(states, params["nn_params"])
    )


def residual_mse_curve_on_true_trajectory(trained, true_trajectories):
    states = np.asarray(true_trajectories, dtype=float)
    n_batch, n_time, state_dim = states.shape
    flat_states = states.reshape(-1, state_dim)
    model_vf = model_rhs_numpy(flat_states, trained["params"])
    true_vf = true_rhs_numpy(flat_states)
    residual_sq = np.sum((model_vf - true_vf) ** 2, axis=1)
    return residual_sq.reshape(n_batch, n_time).mean(axis=0)


def missing_term_curves_on_true_trajectory(trained, true_trajectories):
    states = np.asarray(true_trajectories, dtype=float)
    n_batch, n_time, state_dim = states.shape
    flat_states = states.reshape(-1, state_dim)
    params = trained["params"]
    true_missing = true_rhs_numpy(flat_states) - f_physics_numpy(
        flat_states,
        params["f_physics"],
    )
    learned_missing = nn_numpy(flat_states, params["nn_params"])
    error_sq = np.sum((learned_missing - true_missing) ** 2, axis=1)
    return {
        "error_mse_by_time": error_sq.reshape(n_batch, n_time).mean(axis=0),
        "true_mean_by_time": true_missing.reshape(n_batch, n_time, state_dim).mean(axis=0),
        "learned_mean_by_time": learned_missing.reshape(n_batch, n_time, state_dim).mean(axis=0),
    }


def config_label(row):
    method = row.get("training_method", "")
    ratio = row.get("ratio", "")
    if ratio == "adaptive":
        return f"{method}\nadaptive"
    return f"{method}\nr={ratio}"


def config_sort_key(row_or_key):
    if isinstance(row_or_key, tuple):
        method, ratio = row_or_key
    else:
        method = row_or_key.get("training_method")
        ratio = row_or_key.get("ratio")
    ratio_num = parse_int(ratio)
    ratio_key = ratio_num if ratio_num is not None else 10**9
    return method_sort_key(method), ratio_key


def sorted_configs(rows):
    configs = {
        (row.get("training_method"), row.get("ratio"))
        for row in rows
        if row.get("training_method")
    }
    return sorted(configs, key=config_sort_key)


def subset(rows, method=None, ratio=None, seed=None):
    out = rows
    if method is not None:
        out = [row for row in out if row.get("training_method") == method]
    if ratio is not None:
        out = [row for row in out if row.get("ratio") == str(ratio)]
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


def log_space_mean_and_band(curve_array, eps=1e-16):
    curve_array = np.asarray(curve_array, dtype=float)
    safe_curves = np.maximum(curve_array, eps)
    log_curves = np.log10(safe_curves)
    log_mean = np.mean(log_curves, axis=0)
    log_std = (
        np.std(log_curves, axis=0, ddof=1)
        if log_curves.shape[0] >= 2
        else np.zeros_like(log_mean)
    )
    center = 10**log_mean
    lower = 10 ** (log_mean - log_std)
    upper = 10 ** (log_mean + log_std)
    return center, lower, upper, log_std


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


def write_combined_csv(rows, output_dir):
    path = output_dir / "combined_no_noise_training_solver_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary(rows, output_dir):
    metrics = [
        "best_loss",
        "train_mse",
        "validation_mse",
        "extrapolate_mse",
        "final_mse",
        "residual_mse",
        "model_vf_rel_error",
    ]
    summary_rows = []
    for method, ratio in sorted_configs(rows):
        group = [
            row
            for row in rows
            if row.get("training_method") == method and row.get("ratio") == ratio
        ]
        if not group:
            continue
        summary = {
            "training_method": method,
            "ratio": ratio,
            "evaluation_protocols": ";".join(
                sorted({row.get("evaluation_protocol", "") for row in group})
            ),
            "n_runs": len(group),
            "seeds": ";".join(
                str(seed)
                for seed in sorted(
                    {
                        parse_int(row.get("seed"))
                        for row in group
                        if parse_int(row.get("seed")) is not None
                    }
                )
            ),
        }
        for metric in metrics:
            stats = summarize(metric_values(group, metric))
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summary_rows.append(summary)

    if not summary_rows:
        return
    path = output_dir / "combined_group_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_correlation_summary(rows, output_dir):
    groups = [("all", rows)]
    for method in sorted({row.get("training_method") for row in rows}, key=method_sort_key):
        method_rows = [row for row in rows if row.get("training_method") == method]
        if method_rows:
            groups.append((method, method_rows))

    pairs = [
        ("best_loss", "extrapolate_mse"),
        ("validation_mse", "extrapolate_mse"),
        ("residual_mse", "extrapolate_mse"),
        ("model_vf_rel_error", "extrapolate_mse"),
        ("best_loss", "residual_mse"),
    ]
    out_rows = []
    for group_name, group_rows in groups:
        for x_metric, y_metric in pairs:
            x = [parse_float(row.get(x_metric)) for row in group_rows]
            y = [parse_float(row.get(y_metric)) for row in group_rows]
            out_rows.append(
                {
                    "group": group_name,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": len(metric_values(group_rows, x_metric)),
                    "correlation": correlation(x, y),
                }
            )

    path = output_dir / "combined_correlation_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def plot_metric_by_config(rows, output_dir, metric, ylabel):
    configs = sorted_configs(rows)
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(configs)), 5.5))
    x_positions = np.arange(len(configs))

    for idx, (method, ratio) in enumerate(configs):
        group = [
            row
            for row in rows
            if row.get("training_method") == method and row.get("ratio") == ratio
        ]
        values = metric_values(group, metric)
        if values.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else [0.0]
        ax.scatter(
            np.full(values.shape, x_positions[idx]) + np.asarray(jitter),
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

    labels = [f"{method}\n{ratio}" for method, ratio in configs]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Training Solver / Ratio")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_{metric}_by_training_config.png", dpi=220)
    plt.close(fig)


def plot_scatter(rows, output_dir, x_metric, y_metric, title):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    methods = sorted(
        {row.get("training_method") for row in rows if row.get("training_method")},
        key=method_sort_key,
    )

    for method in methods:
        group = [row for row in rows if row.get("training_method") == method]
        x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
        y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
        ratios = np.array(
            [
                parse_float(row.get("ratio"), default=np.nan)
                for row in group
            ],
            dtype=float,
        )
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if not np.any(mask):
            continue
        label = f"{method}, r={correlation(x, y):.2f}"
        ax.scatter(
            x[mask],
            y[mask],
            s=42,
            alpha=0.8,
            label=label,
        )
        for xi, yi, ratio in zip(x[mask], y[mask], ratios[mask]):
            if np.isfinite(ratio):
                ax.text(xi, yi, str(int(ratio)), fontsize=7, alpha=0.65)
            else:
                ax.text(xi, yi, "A", fontsize=7, alpha=0.65)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_{x_metric}_vs_{y_metric}.png", dpi=220)
    plt.close(fig)


def plot_seed_matched_methods(rows, output_dir, metric):
    seeds = sorted(
        {
            parse_int(row.get("seed"))
            for row in rows
            if parse_int(row.get("seed")) is not None
        }
    )
    methods = sorted(
        {row.get("training_method") for row in rows if row.get("training_method")},
        key=method_sort_key,
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        method_rows = [row for row in rows if row.get("training_method") == method]
        best_by_seed = []
        for seed in seeds:
            seed_rows = [
                row for row in method_rows if parse_int(row.get("seed")) == seed
            ]
            values = metric_values(seed_rows, metric)
            best_by_seed.append(np.min(values) if values.size else np.nan)
        best_by_seed = np.asarray(best_by_seed, dtype=float)
        mask = np.isfinite(best_by_seed) & (best_by_seed > 0.0)
        if np.any(mask):
            ax.plot(
                np.asarray(seeds)[mask],
                best_by_seed[mask],
                marker="o",
                label=method,
            )

    ax.set_yscale("log")
    ax.set_xlabel("seed")
    ax.set_ylabel(f"best {metric} across ratios")
    ax.set_title(f"Seed-Matched Best {metric} by Training Method")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_seed_matched_best_{metric}.png", dpi=220)
    plt.close(fig)


def selected_residual_configs(trained_models, ratios):
    configs = []
    for method in ["forward_euler", "heun", "rk4"]:
        for ratio in ratios:
            if any((method, str(ratio), seed) in trained_models for seed in range(100)):
                configs.append((method, str(ratio)))
    if any(
        (method, ratio, seed) in trained_models
        for (method, ratio, seed) in trained_models
        if method == "diffrax_tsit5" and ratio == "adaptive"
    ):
        configs.append(("diffrax_tsit5", "adaptive"))
    return configs


def write_residual_time_summary(curves_by_config, time_grid, output_dir):
    rows = []
    for (method, ratio), curves in curves_by_config.items():
        curve_array = np.asarray(curves, dtype=float)
        if curve_array.size == 0:
            continue
        mean_curve = np.mean(curve_array, axis=0)
        std_curve = np.std(curve_array, axis=0, ddof=1) if len(curves) >= 2 else np.zeros_like(mean_curve)
        min_curve = np.min(curve_array, axis=0)
        max_curve = np.max(curve_array, axis=0)
        log_center, log_lower, log_upper, log_std = log_space_mean_and_band(curve_array)
        for time, mean, std, min_value, max_value, center, lower, upper, log_sigma in zip(
            time_grid,
            mean_curve,
            std_curve,
            min_curve,
            max_curve,
            log_center,
            log_lower,
            log_upper,
            log_std,
        ):
            rows.append(
                {
                    "training_method": method,
                    "ratio": ratio,
                    "time": float(time),
                    "residual_mse_mean": float(mean),
                    "residual_mse_std": float(std),
                    "residual_mse_min": float(min_value),
                    "residual_mse_max": float(max_value),
                    "residual_mse_log_space_center": float(center),
                    "residual_mse_log_space_lower": float(lower),
                    "residual_mse_log_space_upper": float(upper),
                    "residual_mse_log10_std": float(log_sigma),
                    "n_seeds": len(curves),
                }
            )

    if not rows:
        return
    path = output_dir / "combined_residual_mse_vs_time_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_residual_mse_vs_time(trained_models, output_dir, ratios):
    if not trained_models:
        return

    data = make_reference_data()
    true_trajectories = np.asarray(data["extrap"], dtype=float)
    time_grid = np.asarray(t_extrapolate, dtype=float)
    configs = selected_residual_configs(trained_models, ratios)
    curves_by_config = {}

    for method, ratio in configs:
        curves = []
        seeds = sorted(
            seed
            for model_method, model_ratio, seed in trained_models
            if model_method == method and model_ratio == ratio
        )
        for seed in seeds:
            trained = trained_models[(method, ratio, seed)]
            curves.append(
                residual_mse_curve_on_true_trajectory(trained, true_trajectories)
            )
        if curves:
            curves_by_config[(method, ratio)] = curves

    if not curves_by_config:
        return

    write_residual_time_summary(curves_by_config, time_grid, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, squeeze=False)
    axes_by_method = {
        "forward_euler": axes[0, 0],
        "heun": axes[0, 1],
        "rk4": axes[1, 0],
        "diffrax_tsit5": axes[1, 1],
    }

    for (method, ratio), curves in curves_by_config.items():
        ax = axes_by_method.get(method)
        if ax is None:
            continue
        curve_array = np.asarray(curves, dtype=float)
        center, lower, upper, _ = log_space_mean_and_band(curve_array)
        label = "adaptive" if ratio == "adaptive" else f"ratio={ratio}"
        ax.semilogy(time_grid, center, label=label)
        ax.fill_between(time_grid, lower, upper, alpha=0.18)

    for method, ax in axes_by_method.items():
        ax.axvline(T, color="tab:red", linestyle=":", linewidth=1.2)
        ax.set_title(method)
        ax.set_xlabel("t")
        ax.set_ylabel(r"mean $\|f_\theta(y_{true}(t)) - f_{true}(y_{true}(t))\|^2$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Residual MSE vs Time on True Trajectories\n"
        "line = log-space seed mean, shade = +/-1 log-space std",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "combined_residual_mse_vs_time_by_solver.png", dpi=220)
    plt.close(fig)


def write_missing_term_time_summary(curves_by_config, time_grid, output_dir):
    rows = []
    for (method, ratio), curves in curves_by_config.items():
        curve_array = np.asarray(
            [curve["error_mse_by_time"] for curve in curves],
            dtype=float,
        )
        if curve_array.size == 0:
            continue
        mean_curve = np.mean(curve_array, axis=0)
        std_curve = (
            np.std(curve_array, axis=0, ddof=1)
            if curve_array.shape[0] >= 2
            else np.zeros_like(mean_curve)
        )
        log_center, log_lower, log_upper, log_std = log_space_mean_and_band(curve_array)
        for time, mean, std, center, lower, upper, log_sigma in zip(
            time_grid,
            mean_curve,
            std_curve,
            log_center,
            log_lower,
            log_upper,
            log_std,
        ):
            rows.append(
                {
                    "training_method": method,
                    "ratio": ratio,
                    "time": float(time),
                    "missing_term_error_mse_mean": float(mean),
                    "missing_term_error_mse_std": float(std),
                    "missing_term_error_mse_log_space_center": float(center),
                    "missing_term_error_mse_log_space_lower": float(lower),
                    "missing_term_error_mse_log_space_upper": float(upper),
                    "missing_term_error_mse_log10_std": float(log_sigma),
                    "n_seeds": len(curves),
                }
            )

    if not rows:
        return
    path = output_dir / "combined_missing_term_error_vs_time_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_missing_configs(trained_models):
    candidates = [
        ("forward_euler", "1"),
        ("forward_euler", "16"),
        ("heun", "4"),
        ("rk4", "4"),
        ("diffrax_tsit5", "adaptive"),
    ]
    return [
        config
        for config in candidates
        if any(
            key[0] == config[0] and key[1] == config[1]
            for key in trained_models
        )
    ]


def collect_missing_term_curves(trained_models, true_trajectories, configs):
    curves_by_config = {}
    for method, ratio in configs:
        curves = []
        seeds = sorted(
            seed
            for model_method, model_ratio, seed in trained_models
            if model_method == method and model_ratio == ratio
        )
        for seed in seeds:
            trained = trained_models[(method, ratio, seed)]
            curves.append(
                missing_term_curves_on_true_trajectory(trained, true_trajectories)
            )
        if curves:
            curves_by_config[(method, ratio)] = curves
    return curves_by_config


def plot_missing_term_error_vs_time(trained_models, output_dir, ratios):
    if not trained_models:
        return

    data = make_reference_data()
    true_trajectories = np.asarray(data["extrap"], dtype=float)
    time_grid = np.asarray(t_extrapolate, dtype=float)
    configs = selected_residual_configs(trained_models, ratios)
    curves_by_config = collect_missing_term_curves(
        trained_models,
        true_trajectories,
        configs,
    )
    if not curves_by_config:
        return

    write_missing_term_time_summary(curves_by_config, time_grid, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, squeeze=False)
    axes_by_method = {
        "forward_euler": axes[0, 0],
        "heun": axes[0, 1],
        "rk4": axes[1, 0],
        "diffrax_tsit5": axes[1, 1],
    }
    for (method, ratio), curves in curves_by_config.items():
        ax = axes_by_method.get(method)
        if ax is None:
            continue
        curve_array = np.asarray(
            [curve["error_mse_by_time"] for curve in curves],
            dtype=float,
        )
        center, lower, upper, _ = log_space_mean_and_band(curve_array)
        label = "adaptive" if ratio == "adaptive" else f"ratio={ratio}"
        ax.semilogy(time_grid, center, label=label)
        ax.fill_between(time_grid, lower, upper, alpha=0.18)

    for method, ax in axes_by_method.items():
        ax.axvline(T, color="tab:red", linestyle=":", linewidth=1.2)
        ax.set_title(method)
        ax.set_xlabel("t")
        ax.set_ylabel(r"mean $\|g_{NN}(y_{true}(t))-g_{true}(y_{true}(t))\|^2$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Missing-Term Error vs Time on True Trajectories\n"
        "line = log-space seed mean, shade = +/-1 log-space std",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "combined_missing_term_error_vs_time_by_solver.png", dpi=220)
    plt.close(fig)


def plot_missing_term_components(trained_models, output_dir):
    if not trained_models:
        return

    data = make_reference_data()
    true_trajectories = np.asarray(data["extrap"], dtype=float)
    time_grid = np.asarray(t_extrapolate, dtype=float)
    configs = representative_missing_configs(trained_models)
    curves_by_config = collect_missing_term_curves(
        trained_models,
        true_trajectories,
        configs,
    )
    if not curves_by_config:
        return

    fig, axes = plt.subplots(
        len(curves_by_config),
        2,
        figsize=(12, 2.8 * len(curves_by_config)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, ((method, ratio), curves) in enumerate(curves_by_config.items()):
        true_components = np.asarray(
            [curve["true_mean_by_time"] for curve in curves],
            dtype=float,
        ).mean(axis=0)
        learned_components = np.asarray(
            [curve["learned_mean_by_time"] for curve in curves],
            dtype=float,
        )
        learned_mean = learned_components.mean(axis=0)
        learned_std = (
            learned_components.std(axis=0, ddof=1)
            if learned_components.shape[0] >= 2
            else np.zeros_like(learned_mean)
        )
        config_label_text = f"{method}, ratio={ratio}"
        for component_idx in range(2):
            ax = axes[row_idx, component_idx]
            ax.plot(
                time_grid,
                true_components[:, component_idx],
                color="black",
                linewidth=1.8,
                label="true missing term",
            )
            ax.plot(
                time_grid,
                learned_mean[:, component_idx],
                color="tab:blue",
                linestyle="--",
                linewidth=1.8,
                label="NN missing term",
            )
            ax.fill_between(
                time_grid,
                learned_mean[:, component_idx] - learned_std[:, component_idx],
                learned_mean[:, component_idx] + learned_std[:, component_idx],
                color="tab:blue",
                alpha=0.15,
                label="NN seed std" if row_idx == 0 else None,
            )
            ax.axvline(T, color="tab:red", linestyle=":", linewidth=1.2)
            ax.set_title(f"{config_label_text} | g_{component_idx + 1}(t)")
            ax.set_xlabel("t")
            ax.set_ylabel("missing term value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    fig.suptitle(
        "True Missing Term vs Learned NN Missing Term\n"
        "evaluated on true trajectories; curves averaged over initial conditions and seeds",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "combined_missing_term_components_representative.png", dpi=220)
    plt.close(fig)


def visualize(rows, trained_models, output_dir, residual_ratios):
    plot_metric_by_config(rows, output_dir, "extrapolate_mse", "Extrapolation MSE")
    plot_metric_by_config(rows, output_dir, "final_mse", "Final MSE")
    plot_metric_by_config(rows, output_dir, "residual_mse", "Residual MSE")
    plot_metric_by_config(rows, output_dir, "best_loss", "Best Training Loss")
    plot_scatter(
        rows,
        output_dir,
        "best_loss",
        "extrapolate_mse",
        "Optimization Quality vs Extrapolation",
    )
    plot_scatter(
        rows,
        output_dir,
        "residual_mse",
        "extrapolate_mse",
        "Residual Accuracy vs Extrapolation",
    )
    plot_scatter(
        rows,
        output_dir,
        "validation_mse",
        "extrapolate_mse",
        "Validation Error vs Extrapolation",
    )
    plot_seed_matched_methods(rows, output_dir, "extrapolate_mse")
    plot_seed_matched_methods(rows, output_dir, "residual_mse")
    plot_residual_mse_vs_time(trained_models, output_dir, residual_ratios)
    plot_missing_term_error_vs_time(trained_models, output_dir, residual_ratios)
    plot_missing_term_components(trained_models, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine no-noise solver training summaries and generate comparison plots."
    )
    parser.add_argument(
        "--output-dir",
        default="combined_no_noise_training_solver_analysis",
        help="Directory for combined CSVs and visualizations.",
    )
    parser.add_argument(
        "--residual-ratios",
        nargs="+",
        type=int,
        default=[1, 4, 16],
        help="Train ratios shown in the residual-MSE-vs-time plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sources(script_dir, DEFAULT_SOURCES)
    trained_models = load_trained_models(script_dir, DEFAULT_DETAIL_SOURCES)
    write_combined_csv(rows, output_dir)
    write_group_summary(rows, output_dir)
    write_correlation_summary(rows, output_dir)
    visualize(rows, trained_models, output_dir, args.residual_ratios)
    print(f"Saved combined no-noise analysis to {output_dir}")


if __name__ == "__main__":
    main()
