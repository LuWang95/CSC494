import argparse
import csv
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["forward_euler", "heun", "rk4"]
STATE_LABELS = ["prey", "predator"]
STATE_SCALE = np.array([50.0, 50.0])


def method_sort_key(method_name):
    if method_name in METHOD_ORDER:
        return METHOD_ORDER.index(method_name)
    return len(METHOD_ORDER), method_name


def parse_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
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


def sorted_methods(rows, key):
    return sorted(
        {row[key] for row in rows if row.get(key)},
        key=method_sort_key,
    )


def sorted_int_values(rows, key):
    values = {parse_int(row.get(key)) for row in rows}
    return sorted(value for value in values if value is not None)


def make_time_grid(curves, t_final):
    for curve_set in curves.values():
        if "extrapolate_mse_by_time" in curve_set:
            n_times = len(curve_set["extrapolate_mse_by_time"])
            return np.linspace(0.0, t_final, n_times)
    return np.array([])


def safe_log10(values):
    values = np.asarray(values, dtype=float)
    logged = np.full_like(values, np.nan, dtype=float)
    mask = np.isfinite(values) & (values > 0.0)
    logged[mask] = np.log10(values[mask])
    return logged


def format_cell(value):
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def fill_missing_final_mse(rows, details):
    for row in rows:
        train_ratio = parse_int(row.get("ratio"))
        predict_ratio = parse_int(row.get("predict_ratio"))
        key = (
            row.get("training_method"),
            train_ratio,
            row.get("prediction_method"),
            predict_ratio,
        )
        curves = details.get(key)
        if not curves:
            continue
        final_mse = parse_float(row.get("final_mse"))
        if not np.isfinite(final_mse):
            curve = np.asarray(curves.get("extrapolate_mse_by_time", []), dtype=float)
            if curve.size:
                row["final_mse"] = float(curve[-1])
    return rows


def row_metric(row, metric):
    return parse_float(row.get(metric))


def prediction_key(row):
    return (
        row.get("training_method"),
        parse_int(row.get("ratio")),
        row.get("prediction_method"),
        parse_int(row.get("predict_ratio")),
    )


def trained_key(method_name, ratio):
    return method_name, int(ratio)


def swish(x):
    return x / (1.0 + np.exp(-x))


def nn_residual(states, nn_params):
    activations = np.asarray(states, dtype=float) / STATE_SCALE
    for weights, bias in nn_params[:-1]:
        activations = swish(activations @ np.asarray(weights).T + np.asarray(bias))
    final_weights, final_bias = nn_params[-1]
    return activations @ np.asarray(final_weights).T + np.asarray(final_bias)


def residual_on_grid(trained, states):
    params = trained["params"]
    return nn_residual(states, params["nn_params"])


def residual_grid(prey_bounds, predator_bounds, n_grid):
    prey_values = np.linspace(prey_bounds[0], prey_bounds[1], n_grid)
    predator_values = np.linspace(predator_bounds[0], predator_bounds[1], n_grid)
    prey_grid, predator_grid = np.meshgrid(prey_values, predator_values)
    states = np.stack([prey_grid.reshape(-1), predator_grid.reshape(-1)], axis=1)
    return prey_grid, predator_grid, states


def plot_baseline_order(rows, output_dir):
    if not rows:
        return

    metrics = [
        ("final_l2_error", "Final L2 Error"),
        ("final_mse", "Final MSE"),
        ("extrapolate_mse", "Extrapolation MSE"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8))

    for ax, (metric, label) in zip(axes, metrics):
        for method in sorted_methods(rows, "prediction_method"):
            method_rows = [
                row for row in rows if row.get("prediction_method") == method
            ]
            method_rows.sort(key=lambda row: parse_float(row.get("h_predict")))
            hs = np.array([row_metric(row, "h_predict") for row in method_rows])
            errors = np.array([row_metric(row, metric) for row in method_rows])
            mask = np.isfinite(hs) & np.isfinite(errors) & (hs > 0) & (errors > 0)
            if not np.any(mask):
                continue
            ax.loglog(hs[mask], errors[mask], marker="o", label=method)
            if np.count_nonzero(mask) >= 2:
                slope = np.polyfit(np.log(hs[mask]), np.log(errors[mask]), 1)[0]
                ax.text(hs[mask][-1], errors[mask][-1], f"{slope:.2f}", fontsize=8)
        ax.set_xlabel("prediction step size")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("True RHS Baseline: Error vs Step Size", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_error_order.png", dpi=220)
    plt.close(fig)


def plot_baseline_time_curves(details, output_dir, t_final, extrap_start):
    if not details:
        return

    time_grid = make_time_grid(details, t_final)
    if time_grid.size == 0:
        return

    methods = sorted({key[0] for key in details}, key=method_sort_key)
    fig, axes = plt.subplots(
        len(methods), 2, figsize=(12, 4 * len(methods)), squeeze=False
    )

    for row_idx, method in enumerate(methods):
        method_items = [
            (key, curves) for key, curves in details.items() if key[0] == method
        ]
        method_items.sort(key=lambda item: item[0][1])

        for key, curves in method_items:
            ratio = key[1]
            mse = np.asarray(curves["extrapolate_mse_by_time"], dtype=float)
            rel = np.asarray(curves["extrapolate_rel_error_by_time"], dtype=float)
            axes[row_idx, 0].semilogy(time_grid, mse, label=f"ratio={ratio}")
            axes[row_idx, 1].semilogy(time_grid, rel, label=f"ratio={ratio}")

        for ax, ylabel in zip(
            axes[row_idx],
            ["MSE by time", "Relative error by time"],
        ):
            ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
            ax.set_title(method)
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle("True RHS Baseline: Error Growth Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_error_over_time.png", dpi=220)
    plt.close(fig)


def metric_matrix(rows, train_method, pred_method, train_ratios, predict_ratios, metric):
    matrix = np.full((len(train_ratios), len(predict_ratios)), np.nan)
    for row in rows:
        if (
            row.get("training_method") != train_method
            or row.get("prediction_method") != pred_method
        ):
            continue
        train_ratio = parse_int(row.get("ratio"))
        predict_ratio = parse_int(row.get("predict_ratio"))
        if train_ratio not in train_ratios or predict_ratio not in predict_ratios:
            continue
        i = train_ratios.index(train_ratio)
        j = predict_ratios.index(predict_ratio)
        matrix[i, j] = row_metric(row, metric)
    return matrix


def plot_prediction_heatmaps(rows, output_dir):
    if not rows:
        return

    metrics = [
        ("final_mse", "Final MSE", True),
        ("extrapolate_mse", "Extrapolation MSE", True),
        ("validation_mse", "Validation MSE", True),
        ("model_vf_rel_error", "Vector Field Relative Error", True),
        ("instability_rate", "Instability Rate", False),
    ]
    train_methods = sorted_methods(rows, "training_method")
    pred_methods = sorted_methods(rows, "prediction_method")
    train_ratios = sorted_int_values(rows, "ratio")
    predict_ratios = sorted_int_values(rows, "predict_ratio")

    for metric, title, use_log in metrics:
        raw_matrices = [
            metric_matrix(
                rows,
                train_method,
                pred_method,
                train_ratios,
                predict_ratios,
                metric,
            )
            for train_method in train_methods
            for pred_method in pred_methods
        ]
        values = np.concatenate([matrix.ravel() for matrix in raw_matrices])
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        if use_log:
            plotted_values = safe_log10(values)
            finite_plotted = plotted_values[np.isfinite(plotted_values)]
            if finite_plotted.size == 0:
                continue
            vmin, vmax = finite_plotted.min(), finite_plotted.max()
            color_label = f"log10({title})"
        else:
            vmin, vmax = values.min(), values.max()
            color_label = title

        fig, axes = plt.subplots(
            len(train_methods),
            len(pred_methods),
            figsize=(4.2 * len(pred_methods), 3.8 * len(train_methods)),
            squeeze=False,
        )

        for i, train_method in enumerate(train_methods):
            for j, pred_method in enumerate(pred_methods):
                ax = axes[i, j]
                matrix = metric_matrix(
                    rows,
                    train_method,
                    pred_method,
                    train_ratios,
                    predict_ratios,
                    metric,
                )
                image_data = safe_log10(matrix) if use_log else matrix
                im = ax.imshow(
                    image_data,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
                ax.set_xticks(np.arange(len(predict_ratios)))
                ax.set_xticklabels(predict_ratios)
                ax.set_yticks(np.arange(len(train_ratios)))
                ax.set_yticklabels(train_ratios)
                ax.set_xlabel("predict ratio")
                ax.set_ylabel("train ratio")
                ax.set_title(f"train={train_method}\npred={pred_method}")

                for row_idx in range(matrix.shape[0]):
                    for col_idx in range(matrix.shape[1]):
                        ax.text(
                            col_idx,
                            row_idx,
                            format_cell(matrix[row_idx, col_idx]),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                        )

        fig.suptitle(f"Prediction Sweep: {title}", fontsize=15)
        fig.subplots_adjust(right=0.9, top=0.9, wspace=0.35, hspace=0.45)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=color_label)
        filename = f"prediction_heatmap_{metric}.png"
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def plot_prediction_step_size_curves(rows, output_dir):
    if not rows:
        return

    train_methods = sorted_methods(rows, "training_method")
    pred_methods = sorted_methods(rows, "prediction_method")
    train_ratios = sorted_int_values(rows, "ratio")
    metrics = [
        ("final_mse", "Final MSE"),
        ("extrapolate_mse", "Extrapolation MSE"),
    ]

    for metric, label in metrics:
        for train_method in train_methods:
            fig, axes = plt.subplots(
                1, len(pred_methods), figsize=(5 * len(pred_methods), 4.2), squeeze=False
            )
            for ax, pred_method in zip(axes[0], pred_methods):
                for train_ratio in train_ratios:
                    subset = [
                        row
                        for row in rows
                        if row.get("training_method") == train_method
                        and row.get("prediction_method") == pred_method
                        and parse_int(row.get("ratio")) == train_ratio
                    ]
                    subset.sort(key=lambda row: row_metric(row, "h_predict"))
                    hs = np.array([row_metric(row, "h_predict") for row in subset])
                    errors = np.array([row_metric(row, metric) for row in subset])
                    mask = (
                        np.isfinite(hs)
                        & np.isfinite(errors)
                        & (hs > 0)
                        & (errors > 0)
                    )
                    if np.any(mask):
                        ax.loglog(
                            hs[mask],
                            errors[mask],
                            marker="o",
                            label=f"train_ratio={train_ratio}",
                        )
                ax.set_title(f"pred={pred_method}")
                ax.set_xlabel("prediction step size")
                ax.set_ylabel(label)
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=7)

            fig.suptitle(f"{label} vs Step Size | train={train_method}", fontsize=14)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"prediction_{metric}_vs_stepsize_train_{train_method}.png",
                dpi=220,
            )
            plt.close(fig)


def unique_training_metric(rows, metric):
    values = {}
    for row in rows:
        method = row.get("training_method")
        ratio = parse_int(row.get("ratio"))
        value = row_metric(row, metric)
        if not method or ratio is None or not np.isfinite(value):
            continue
        values.setdefault((method, ratio), []).append(value)
    return {
        key: float(np.mean(metric_values))
        for key, metric_values in values.items()
        if metric_values
    }


def plot_residual_mse_vs_train_ratio(rows, output_dir):
    residual_mse = unique_training_metric(rows, "residual_mse")
    if not residual_mse:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in sorted({key[0] for key in residual_mse}, key=method_sort_key):
        method_items = [
            (ratio, value)
            for (method_name, ratio), value in residual_mse.items()
            if method_name == method
        ]
        method_items.sort()
        ratios = np.array([item[0] for item in method_items], dtype=float)
        values = np.array([item[1] for item in method_items], dtype=float)
        mask = np.isfinite(values) & (values > 0.0)
        if np.any(mask):
            ax.loglog(ratios[mask], values[mask], marker="o", label=method)

    ax.set_xlabel("train ratio")
    ax.set_ylabel("Residual MSE")
    ax.set_title("Residual MSE vs Training Step Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_mse_vs_train_ratio.png", dpi=220)
    plt.close(fig)


def plot_residual_heatmap(rows, output_dir):
    if not rows:
        return

    train_methods = sorted_methods(rows, "training_method")
    pred_methods = sorted_methods(rows, "prediction_method")
    train_ratios = sorted_int_values(rows, "ratio")
    predict_ratios = sorted_int_values(rows, "predict_ratio")
    matrices = [
        metric_matrix(
            rows,
            train_method,
            pred_method,
            train_ratios,
            predict_ratios,
            "residual_mse",
        )
        for train_method in train_methods
        for pred_method in pred_methods
    ]
    values = np.concatenate([matrix.ravel() for matrix in matrices])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    log_values = safe_log10(values)
    log_values = log_values[np.isfinite(log_values)]
    if log_values.size == 0:
        return

    fig, axes = plt.subplots(
        len(train_methods),
        len(pred_methods),
        figsize=(4.2 * len(pred_methods), 3.8 * len(train_methods)),
        squeeze=False,
    )
    for i, train_method in enumerate(train_methods):
        for j, pred_method in enumerate(pred_methods):
            ax = axes[i, j]
            matrix = metric_matrix(
                rows,
                train_method,
                pred_method,
                train_ratios,
                predict_ratios,
                "residual_mse",
            )
            im = ax.imshow(
                safe_log10(matrix),
                aspect="auto",
                origin="lower",
                vmin=log_values.min(),
                vmax=log_values.max(),
                cmap="viridis",
            )
            ax.set_xticks(np.arange(len(predict_ratios)))
            ax.set_xticklabels(predict_ratios)
            ax.set_yticks(np.arange(len(train_ratios)))
            ax.set_yticklabels(train_ratios)
            ax.set_xlabel("predict ratio")
            ax.set_ylabel("train ratio")
            ax.set_title(f"train={train_method}\npred={pred_method}")
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    ax.text(
                        col_idx,
                        row_idx,
                        format_cell(matrix[row_idx, col_idx]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )

    fig.suptitle("Prediction Sweep: Residual MSE", fontsize=15)
    fig.subplots_adjust(right=0.9, top=0.9, wspace=0.35, hspace=0.45)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="log10(Residual MSE)")
    fig.savefig(output_dir / "prediction_heatmap_residual_mse.png", dpi=220)
    plt.close(fig)


def plot_residual_difference(trained, output_dir, prey_bounds, predator_bounds, n_grid):
    if not trained:
        return

    _, _, states = residual_grid(prey_bounds, predator_bounds, n_grid)
    methods = sorted({key[0] for key in trained}, key=method_sort_key)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    wrote_anything = False

    for method in methods:
        ratios = sorted(ratio for method_name, ratio in trained if method_name == method)
        transitions = []
        differences = []
        coarse_steps = []
        for coarse_ratio, fine_ratio in zip(ratios[:-1], ratios[1:]):
            if fine_ratio != 2 * coarse_ratio:
                continue
            coarse = residual_on_grid(trained[trained_key(method, coarse_ratio)], states)
            fine = residual_on_grid(trained[trained_key(method, fine_ratio)], states)
            differences.append(float(np.mean((coarse - fine) ** 2)))
            coarse_steps.append(1.0 / coarse_ratio)
            transitions.append(f"{coarse_ratio}->{fine_ratio}")

        differences = np.array(differences, dtype=float)
        coarse_steps = np.array(coarse_steps, dtype=float)
        mask = np.isfinite(differences) & (differences > 0.0)
        if not np.any(mask):
            continue
        wrote_anything = True
        ax.loglog(coarse_steps[mask], differences[mask], marker="o", label=method)
        for x, y, label in zip(coarse_steps[mask], differences[mask], np.array(transitions)[mask]):
            ax.text(x, y, label, fontsize=8)

    if not wrote_anything:
        plt.close(fig)
        return

    ax.invert_xaxis()
    ax.set_xlabel("coarse training step size, proportional to 1 / train_ratio")
    ax.set_ylabel(r"mean $\|r_h - r_{h/2}\|^2$")
    ax.set_title("Residual Difference Between Adjacent Train Ratios")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_difference_vs_train_stepsize.png", dpi=220)
    plt.close(fig)


def plot_residual_vector_fields(
    trained,
    output_dir,
    vector_ratios,
    prey_bounds,
    predator_bounds,
    n_grid,
):
    if not trained:
        return

    prey_grid, predator_grid, states = residual_grid(prey_bounds, predator_bounds, n_grid)
    methods = sorted({key[0] for key in trained}, key=method_sort_key)

    for method in methods:
        available_ratios = sorted(
            ratio for method_name, ratio in trained if method_name == method
        )
        ratios = [ratio for ratio in vector_ratios if ratio in available_ratios]
        if not ratios:
            ratios = available_ratios[: min(3, len(available_ratios))]
        if not ratios:
            continue

        residuals = [
            residual_on_grid(trained[trained_key(method, ratio)], states)
            for ratio in ratios
        ]
        magnitudes = [
            np.linalg.norm(residual, axis=1)
            for residual in residuals
        ]
        finite_magnitudes = np.concatenate(
            [mag[np.isfinite(mag)] for mag in magnitudes if np.any(np.isfinite(mag))]
        )
        vmax = finite_magnitudes.max() if finite_magnitudes.size else None

        fig, axes = plt.subplots(
            1,
            len(ratios),
            figsize=(4.8 * len(ratios), 4.4),
            squeeze=False,
        )
        for ax, ratio, residual, magnitude in zip(axes[0], ratios, residuals, magnitudes):
            u = residual[:, 0].reshape(prey_grid.shape)
            v = residual[:, 1].reshape(predator_grid.shape)
            colors = magnitude.reshape(prey_grid.shape)
            quiver = ax.quiver(
                prey_grid,
                predator_grid,
                u,
                v,
                colors,
                angles="xy",
                scale_units="xy",
                scale=None,
                cmap="viridis",
            )
            if vmax is not None:
                quiver.set_clim(0.0, vmax)
            ax.set_xlabel("prey")
            ax.set_ylabel("predator")
            ax.set_title(f"train_ratio={ratio}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

        fig.suptitle(f"Learned Residual Vector Field | train={method}", fontsize=14)
        fig.subplots_adjust(right=0.9, top=0.82, wspace=0.35)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(quiver, cax=cbar_ax, label="residual magnitude")
        fig.savefig(output_dir / f"residual_vector_field_train_{method}.png", dpi=220)
        plt.close(fig)


def plot_generalization_scatter(rows, output_dir):
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    markers = {"forward_euler": "o", "heun": "s", "rk4": "^"}

    for pred_method in sorted_methods(rows, "prediction_method"):
        subset = [row for row in rows if row.get("prediction_method") == pred_method]
        for train_method in sorted_methods(subset, "training_method"):
            train_subset = [
                row for row in subset if row.get("training_method") == train_method
            ]
            x = np.array([row_metric(row, "validation_mse") for row in train_subset])
            y = np.array([row_metric(row, "extrapolate_mse") for row in train_subset])
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    label=f"train={train_method}, pred={pred_method}",
                    marker=markers.get(train_method, "o"),
                    alpha=0.75,
                    s=36,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("validation MSE")
    ax.set_ylabel("extrapolation MSE")
    ax.set_title("Validation Error vs Extrapolation Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_validation_vs_extrapolation.png", dpi=220)
    plt.close(fig)


def plot_best_time_curves(rows, details, output_dir, t_final, extrap_start, top_n):
    if not rows or not details:
        return

    ranked = [
        row
        for row in rows
        if np.isfinite(row_metric(row, "final_mse"))
        and prediction_key(row) in details
    ]
    ranked.sort(key=lambda row: row_metric(row, "final_mse"))
    if not ranked:
        return

    time_grid = make_time_grid(details, t_final)
    if time_grid.size == 0:
        return

    selected = ranked[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in selected:
        key = prediction_key(row)
        curves = details[key]
        mse = np.asarray(curves["extrapolate_mse_by_time"], dtype=float)
        label = (
            f"train={key[0]}:{key[1]}, pred={key[2]}:{key[3]}, "
            f"final={row_metric(row, 'final_mse'):.2e}"
        )
        ax.semilogy(time_grid, mse, label=label)

    ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE by time")
    ax.set_title(f"Top {len(selected)} Prediction Configurations by Final MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_top_configs_error_over_time.png", dpi=220)
    plt.close(fig)

    best_row = ranked[0]
    best_key = prediction_key(best_row)
    best_curves = details[best_key]
    per_state = np.asarray(best_curves.get("extrapolate_mse_by_time_state", []))
    if per_state.ndim != 2 or per_state.shape[1] == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for state_idx in range(per_state.shape[1]):
        label = STATE_LABELS[state_idx] if state_idx < len(STATE_LABELS) else f"state {state_idx}"
        ax.semilogy(time_grid, per_state[:, state_idx], label=label)
    ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE by time")
    ax.set_title(
        "Best Configuration Per-State Error\n"
        f"train={best_key[0]}:{best_key[1]}, pred={best_key[2]}:{best_key[3]}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_best_config_per_state_error.png", dpi=220)
    plt.close(fig)


def write_top_configurations(rows, output_dir, top_n):
    ranked = [row for row in rows if np.isfinite(row_metric(row, "final_mse"))]
    ranked.sort(key=lambda row: row_metric(row, "final_mse"))
    if not ranked:
        return

    columns = [
        "training_method",
        "ratio",
        "prediction_method",
        "predict_ratio",
        "h_model",
        "h_predict",
        "validation_mse",
        "extrapolate_mse",
        "final_mse",
        "final_l2_error",
        "model_vf_rel_error",
        "instability_rate",
    ]
    with (output_dir / "top_prediction_configurations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked[:top_n])


def visualize_results(args):
    script_dir = Path(__file__).resolve().parent
    baseline_dir = Path(args.baseline_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)

    if not baseline_dir.is_absolute():
        baseline_dir = script_dir / baseline_dir
    if not prediction_dir.is_absolute():
        prediction_dir = script_dir / prediction_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv_rows(baseline_dir / "rhs_baseline_summary.csv")
    baseline_payload = load_pickle(baseline_dir / "rhs_baseline_details.pkl")
    baseline_details = baseline_payload.get("details", {})

    prediction_rows = read_csv_rows(prediction_dir / "prediction_sweep_summary.csv")
    prediction_payload = load_pickle(prediction_dir / "prediction_sweep_details.pkl")
    prediction_details = prediction_payload.get("details", {})
    trained_models = prediction_payload.get("trained", {})
    if not prediction_rows:
        prediction_rows = prediction_payload.get("rows", [])
    prediction_rows = fill_missing_final_mse(prediction_rows, prediction_details)

    plot_baseline_order(baseline_rows, output_dir)
    plot_baseline_time_curves(
        baseline_details,
        output_dir,
        t_final=args.t_final,
        extrap_start=args.extrap_start,
    )
    plot_prediction_heatmaps(prediction_rows, output_dir)
    plot_residual_heatmap(prediction_rows, output_dir)
    plot_residual_mse_vs_train_ratio(prediction_rows, output_dir)
    plot_residual_difference(
        trained_models,
        output_dir,
        prey_bounds=args.prey_bounds,
        predator_bounds=args.predator_bounds,
        n_grid=args.residual_grid_size,
    )
    plot_residual_vector_fields(
        trained_models,
        output_dir,
        vector_ratios=args.residual_vector_ratios,
        prey_bounds=args.prey_bounds,
        predator_bounds=args.predator_bounds,
        n_grid=args.residual_vector_grid_size,
    )
    plot_prediction_step_size_curves(prediction_rows, output_dir)
    plot_generalization_scatter(prediction_rows, output_dir)
    plot_best_time_curves(
        prediction_rows,
        prediction_details,
        output_dir,
        t_final=args.t_final,
        extrap_start=args.extrap_start,
        top_n=args.top_n,
    )
    write_top_configurations(prediction_rows, output_dir, top_n=args.top_n)

    print(f"Saved detailed visualizations to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create detailed visualizations for solver_error_experiment outputs."
    )
    parser.add_argument(
        "--baseline-dir",
        default="rhs_baseline_results",
        help="Directory containing rhs_baseline_summary.csv and rhs_baseline_details.pkl.",
    )
    parser.add_argument(
        "--prediction-dir",
        default="prediction_sweep_results",
        help="Directory containing prediction_sweep_summary.csv and prediction_sweep_details.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default="solver_error_visualizations",
        help="Directory where visualization files will be written.",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=20.0,
        help="Final time used to reconstruct the time axis for saved error curves.",
    )
    parser.add_argument(
        "--extrap-start",
        type=float,
        default=5.0,
        help="Time where extrapolation starts, shown as a vertical line.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of best configurations to show in ranking plots and CSV.",
    )
    parser.add_argument(
        "--residual-vector-ratios",
        nargs="+",
        type=int,
        default=[1, 4, 16],
        help="Train ratios shown in residual vector field plots.",
    )
    parser.add_argument(
        "--residual-grid-size",
        type=int,
        default=30,
        help="Grid size used for residual difference calculations.",
    )
    parser.add_argument(
        "--residual-vector-grid-size",
        type=int,
        default=16,
        help="Grid size used for residual vector field quiver plots.",
    )
    parser.add_argument(
        "--prey-bounds",
        nargs=2,
        type=float,
        default=[5.0, 25.0],
        help="Prey bounds for residual grid visualizations.",
    )
    parser.add_argument(
        "--predator-bounds",
        nargs=2,
        type=float,
        default=[10.0, 40.0],
        help="Predator bounds for residual grid visualizations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize_results(parse_args())
