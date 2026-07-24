"""Visualize the common-RK4 solver-path-dependence experiment.

Reads the CSV outputs of ``solver_path_dependence.py`` and emphasizes the three
questions required for a valid path-dependence claim:

1. Did all branches optimize the same RK4 objective?
2. Did they reach the declared approximate-stationarity thresholds?
3. After stationarity, do their learned residual/vector fields still differ?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SOLVER_ORDER = ["forward_euler", "heun", "rk4"]
SOLVER_LABELS = {
    "forward_euler": "Euler early",
    "heun": "Heun early",
    "rk4": "RK4 throughout",
}
SOLVER_COLORS = {
    "forward_euler": "#1f77b4",
    "heun": "#ff7f0e",
    "rk4": "#2ca02c",
}
SOLVER_MARKERS = {
    "forward_euler": "o",
    "heun": "s",
    "rk4": "^",
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=0):
    number = parse_float(value)
    return int(number) if np.isfinite(number) else default


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_values(rows, field):
    values = np.asarray([parse_float(row.get(field)) for row in rows])
    return values[np.isfinite(values)]


def solver_names(rows):
    present = {row["early_solver"] for row in rows}
    return [name for name in SOLVER_ORDER if name in present]


def grouped(rows, field):
    output = defaultdict(list)
    for row in rows:
        value = parse_float(row.get(field))
        if np.isfinite(value):
            output[row["early_solver"]].append(value)
    return output


def positive(value, floor=1e-30):
    return max(float(value), floor)


def group_summary(rows):
    output = []
    for solver_name in solver_names(rows):
        selected = [row for row in rows if row["early_solver"] == solver_name]
        summary = {
            "early_solver": solver_name,
            "n": len(selected),
            "n_stationary": sum(parse_bool(row.get("stationary")) for row in selected),
            "stationary_fraction": (
                sum(parse_bool(row.get("stationary")) for row in selected)
                / max(len(selected), 1)
            ),
        }
        for field in [
            "switch_common_objective",
            "final_common_objective",
            "final_gradient_rms",
            "final_gradient_max_abs",
            "model_vf_rel_error",
            "residual_rel_error",
            "continuous_extrapolation_mse",
            "residual_rms_distance_to_rk4_path",
            "residual_relative_distance_to_rk4_path",
        ]:
            values = finite_values(selected, field)
            summary[f"{field}_median"] = (
                float(np.median(values)) if values.size else np.nan
            )
            summary[f"{field}_min"] = (
                float(np.min(values)) if values.size else np.nan
            )
            summary[f"{field}_max"] = (
                float(np.max(values)) if values.size else np.nan
            )
        output.append(summary)
    return output


def pairwise_summary(rows):
    output = []
    pair_names = sorted(
        {
            (row["left_early_solver"], row["right_early_solver"])
            for row in rows
        }
    )
    for left, right in pair_names:
        selected = [
            row
            for row in rows
            if row["left_early_solver"] == left
            and row["right_early_solver"] == right
        ]
        values = finite_values(
            selected,
            "residual_difference_relative_to_true_residual",
        )
        output.append(
            {
                "left_early_solver": left,
                "right_early_solver": right,
                "n": len(selected),
                "n_both_stationary": sum(
                    parse_bool(row.get("both_stationary")) for row in selected
                ),
                "n_common_objective_close": sum(
                    parse_bool(row.get("common_objective_close"))
                    for row in selected
                ),
                "n_persistent_functional_path_effect": sum(
                    parse_bool(row.get("persistent_functional_path_effect"))
                    for row in selected
                ),
                "functional_difference_median": (
                    float(np.median(values)) if values.size else np.nan
                ),
                "functional_difference_min": (
                    float(np.min(values)) if values.size else np.nan
                ),
                "functional_difference_max": (
                    float(np.max(values)) if values.size else np.nan
                ),
            }
        )
    return output


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def common_history_rows(history):
    return [
        row
        for row in history
        if row.get("phase")
        in {"common_rk4_switch", "common_rk4_adam", "common_rk4_lbfgs"}
    ]


def history_sort_key(row):
    phase_rank = {
        "common_rk4_switch": 0,
        "common_rk4_adam": 1,
        "common_rk4_lbfgs": 2,
    }
    return (
        parse_int(row.get("common_epoch")),
        phase_rank.get(row.get("phase"), 1),
    )


def plot_common_refinement(history, summary, output_dir):
    selected_history = common_history_rows(history)
    seeds = sorted({parse_int(row.get("seed")) for row in selected_history})
    if not seeds:
        return

    threshold_rms = finite_values(summary, "stationary_grad_rms_threshold")
    grad_threshold = (
        float(threshold_rms[0]) if threshold_rms.size else np.nan
    )
    figure, axes = plt.subplots(
        2,
        len(seeds),
        figsize=(max(7.5, 4.2 * len(seeds)), 7.2),
        squeeze=False,
        sharex="col",
    )

    for column, seed in enumerate(seeds):
        for solver_name in solver_names(summary):
            rows = [
                row
                for row in selected_history
                if parse_int(row.get("seed")) == seed
                and row.get("early_solver") == solver_name
            ]
            rows.sort(key=history_sort_key)
            if not rows:
                continue
            x = [parse_int(row.get("common_epoch")) for row in rows]
            objective = [positive(parse_float(row.get("objective"))) for row in rows]
            gradient = [
                positive(parse_float(row.get("gradient_rms"))) for row in rows
            ]
            axes[0, column].plot(
                x,
                objective,
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                markersize=3,
                linewidth=1.5,
                label=SOLVER_LABELS[solver_name],
            )
            axes[1, column].plot(
                x,
                gradient,
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                markersize=3,
                linewidth=1.5,
                label=SOLVER_LABELS[solver_name],
            )

        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[0, column].set_title(f"seed {seed}")
        axes[1, column].set_xlabel("common RK4 Adam epoch")
        axes[0, column].grid(True, which="both", alpha=0.25)
        axes[1, column].grid(True, which="both", alpha=0.25)
        if np.isfinite(grad_threshold):
            axes[1, column].axhline(
                grad_threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.65,
            )

    axes[0, 0].set_ylabel("common RK4 objective")
    axes[1, 0].set_ylabel("gradient RMS")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Common-objective convergence: do early-solver paths merge?",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "fig01_common_rk4_objective_and_gradient_history.png",
        dpi=220,
    )
    plt.close(figure)


def plot_stationarity(summary, output_dir):
    grad_rms_thresholds = finite_values(
        summary,
        "stationary_grad_rms_threshold",
    )
    grad_max_thresholds = finite_values(
        summary,
        "stationary_grad_max_threshold",
    )
    rms_threshold = (
        float(grad_rms_thresholds[0]) if grad_rms_thresholds.size else np.nan
    )
    max_threshold = (
        float(grad_max_thresholds[0]) if grad_max_thresholds.size else np.nan
    )

    figure, ax = plt.subplots(figsize=(7.2, 5.7))
    for solver_name in solver_names(summary):
        selected = [
            row for row in summary if row["early_solver"] == solver_name
        ]
        for row in selected:
            stationary = parse_bool(row.get("stationary"))
            ax.scatter(
                positive(parse_float(row.get("final_gradient_rms"))),
                positive(parse_float(row.get("final_gradient_max_abs"))),
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name] if stationary else "x",
                s=58,
                alpha=0.9,
                label=(
                    SOLVER_LABELS[solver_name]
                    if not any(
                        collection.get_label() == SOLVER_LABELS[solver_name]
                        for collection in ax.collections
                    )
                    else "_nolegend_"
                ),
            )
            ax.annotate(
                f"s{parse_int(row.get('seed'))}",
                (
                    positive(parse_float(row.get("final_gradient_rms"))),
                    positive(parse_float(row.get("final_gradient_max_abs"))),
                ),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    if np.isfinite(rms_threshold):
        ax.axvline(
            rms_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label="stationarity thresholds",
        )
    if np.isfinite(max_threshold):
        ax.axhline(
            max_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("final gradient RMS")
    ax.set_ylabel("final maximum |gradient component|")
    ax.set_title("Stationarity audit under the common RK4 objective")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig02_stationarity_audit.png", dpi=220)
    plt.close(figure)


def grouped_scatter(
    ax,
    rows,
    field,
    ylabel,
    log_scale=True,
    allow_missing=False,
):
    names = solver_names(rows)
    rng = np.random.default_rng(494)
    for position, solver_name in enumerate(names):
        values = finite_values(
            [row for row in rows if row["early_solver"] == solver_name],
            field,
        )
        if values.size == 0:
            if allow_missing:
                continue
            values = np.asarray([np.nan])
        jitter = rng.uniform(-0.07, 0.07, size=values.size)
        plotted = np.maximum(values, 1e-30) if log_scale else values
        ax.scatter(
            position + jitter,
            plotted,
            color=SOLVER_COLORS[solver_name],
            marker=SOLVER_MARKERS[solver_name],
            alpha=0.75,
            s=36,
        )
        if np.isfinite(values).any():
            median = float(np.nanmedian(values))
            ax.plot(
                position,
                positive(median) if log_scale else median,
                marker="_",
                markersize=18,
                color="black",
            )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SOLVER_LABELS[name] for name in names], rotation=15)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def plot_final_metrics(summary, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    specifications = [
        (
            "final_common_objective",
            "final common RK4 objective",
            "Same final objective",
            False,
        ),
        (
            "model_vf_rel_error",
            "model vector-field relative error",
            "Recovered continuous dynamics",
            False,
        ),
        (
            "continuous_extrapolation_mse",
            "continuous extrapolation MSE",
            "Long-horizon consequence",
            False,
        ),
        (
            "residual_rms_distance_to_rk4_path",
            "residual RMS distance to RK4-history branch",
            "Persistent functional path difference",
            True,
        ),
    ]
    for ax, (field, ylabel, title, allow_missing) in zip(
        axes.reshape(-1),
        specifications,
    ):
        grouped_scatter(
            ax,
            summary,
            field,
            ylabel,
            log_scale=True,
            allow_missing=allow_missing,
        )
        ax.set_title(title)
    figure.suptitle(
        "Final endpoints after identical RK4 Adam + L-BFGS refinement",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "fig03_final_endpoint_metrics.png", dpi=220)
    plt.close(figure)


def pair_label(row):
    left = SOLVER_LABELS.get(row["left_early_solver"], row["left_early_solver"])
    right = SOLVER_LABELS.get(
        row["right_early_solver"],
        row["right_early_solver"],
    )
    return f"{left}\nvs {right}"


def plot_pairwise_effects(pairwise, metadata, output_dir):
    if not pairwise:
        return
    pairs = sorted({pair_label(row) for row in pairwise})
    threshold = parse_float(
        metadata.get("path_effect_decision", {}).get(
            "functional_difference_threshold_relative_to_true_residual"
        )
    )
    figure, ax = plt.subplots(figsize=(max(8.0, 2.2 * len(pairs)), 5.7))
    rng = np.random.default_rng(495)
    for position, label in enumerate(pairs):
        selected = [row for row in pairwise if pair_label(row) == label]
        for row in selected:
            both_stationary = parse_bool(row.get("both_stationary"))
            persistent = parse_bool(
                row.get("persistent_functional_path_effect")
            )
            color = (
                "#d62728"
                if persistent
                else ("#2ca02c" if both_stationary else "#7f7f7f")
            )
            marker = "o" if both_stationary else "x"
            value = positive(
                parse_float(
                    row.get(
                        "residual_difference_relative_to_true_residual"
                    )
                )
            )
            ax.scatter(
                position + rng.uniform(-0.06, 0.06),
                value,
                color=color,
                marker=marker,
                s=52,
                alpha=0.85,
            )
            ax.annotate(
                f"s{parse_int(row.get('seed'))}",
                (position, value),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    if np.isfinite(threshold):
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label="functional-difference threshold",
        )
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs)
    ax.set_yscale("log")
    ax.set_ylabel(
        "RMS residual difference / RMS true missing residual"
    )
    ax.set_title(
        "Pairwise path effect: only stationary pairs support a conclusion"
    )
    ax.grid(True, which="both", alpha=0.25)
    if np.isfinite(threshold):
        ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "fig04_pairwise_functional_path_effect.png", dpi=220)
    plt.close(figure)


def plot_switch_to_final(summary, output_dir):
    seeds = sorted({parse_int(row.get("seed")) for row in summary})
    figure, axes = plt.subplots(
        1,
        len(seeds),
        figsize=(max(7.5, 4.0 * len(seeds)), 4.8),
        squeeze=False,
        sharey=True,
    )
    for column, seed in enumerate(seeds):
        ax = axes[0, column]
        for row in [
            item for item in summary if parse_int(item.get("seed")) == seed
        ]:
            solver_name = row["early_solver"]
            start = positive(parse_float(row.get("switch_common_objective")))
            final = positive(parse_float(row.get("final_common_objective")))
            ax.plot(
                [0, 1],
                [start, final],
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                linewidth=1.8,
                label=SOLVER_LABELS[solver_name],
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["at RK4 switch", "after refinement"])
        ax.set_yscale("log")
        ax.set_title(f"seed {seed}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("common RK4 objective")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("How much of the early-solver separation survives refinement?")
    figure.tight_layout()
    figure.savefig(output_dir / "fig05_switch_to_final_objective.png", dpi=220)
    plt.close(figure)


def visualize(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(input_dir / "summary.csv")
    history = read_csv(input_dir / "training_history.csv")
    pairwise = read_csv(input_dir / "pairwise_path_comparisons.csv")
    metadata_path = input_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as handle:
            metadata = json.load(handle)

    write_csv(group_summary(summary), output_dir / "group_summary.csv")
    write_csv(
        pairwise_summary(pairwise),
        output_dir / "pairwise_decision_summary.csv",
    )
    plot_common_refinement(history, summary, output_dir)
    plot_stationarity(summary, output_dir)
    plot_final_metrics(summary, output_dir)
    plot_pairwise_effects(pairwise, metadata, output_dir)
    plot_switch_to_final(summary, output_dir)
    print(f"[done] visualizations written to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "solver_path_dependence_refined_results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "solver_path_dependence_visualizations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    visualize(arguments.input_dir, arguments.output_dir)
