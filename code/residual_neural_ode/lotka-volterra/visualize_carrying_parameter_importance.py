import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from analyze_carrying_parameter_importance import (
    OUTCOMES,
    interval_regression,
    outcome_arrays,
    regularization_dummies,
    residualize,
    summarize_by_interval,
    summarize_interactions,
    write_rows,
)


INTERVAL_ORDER = [1.25, 2.5, 5.0, 7.5]
REG_ORDER = ["none", "l2_small", "ortho_small", "l2_plus_ortho"]
REG_LABELS = {
    "none": "none",
    "l2_small": "L2",
    "ortho_small": "orthogonality",
    "l2_plus_ortho": "L2 + orthogonality",
}
REG_COLORS = {
    "none": "C0",
    "l2_small": "C1",
    "ortho_small": "C2",
    "l2_plus_ortho": "C3",
}
REG_MARKERS = {
    "none": "o",
    "l2_small": "s",
    "ortho_small": "^",
    "l2_plus_ortho": "D",
}
HORIZONS = [("h2p5", "H=2.5"), ("h5", "H=5"), ("h10", "H=10")]


def parse_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def successful_rows(rows):
    return [row for row in rows if row.get("evaluation_status") == "ok"]


def sorted_intervals(rows):
    present = {parse_float(row.get("train_interval")) for row in rows}
    return [value for value in INTERVAL_ORDER if value in present] + sorted(
        present - set(INTERVAL_ORDER)
    )


def subset_interval(rows, train_interval):
    return [
        row
        for row in rows
        if parse_float(row.get("train_interval")) == train_interval
    ]


def validate_single_protocol(rows):
    signatures = {
        (
            row.get("training_protocol_version"),
            row.get("chunk_size"),
            row.get("stage_scale"),
            row.get("ratio"),
            row.get("noise_level"),
            row.get("coverage_radius"),
        )
        for row in rows
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "Input mixes multiple training/evaluation protocols; visualize them separately."
        )


def profile_values(rows, mask):
    return np.asarray(
        [
            row.get("regularization_profile", "none")
            for row, keep in zip(rows, mask)
            if keep
        ]
    )


def parameter_beta(rows, outcome):
    parameter, vf, y, mask = outcome_arrays(rows, outcome)
    if y.size < 4:
        return np.nan
    profiles = profile_values(rows, mask)
    beta, _, _ = interval_regression(parameter, vf, y, profiles)
    return beta


def bootstrap_parameter_beta(rows, outcome, samples, rng):
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 2:
        return np.nan, np.nan
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample_rows = []
        for seed in sampled:
            sample_rows.extend(row for row in rows if int(row["seed"]) == int(seed))
        estimate = parameter_beta(sample_rows, outcome)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def plot_parameter_effect(rows, output_dir, bootstrap_samples, bootstrap_seed):
    intervals = sorted_intervals(rows)
    rng = np.random.default_rng(bootstrap_seed)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panel_specs = [
        ("same_ic_extrap_mse_", "Same IC"),
        ("candidate_ic_extrap_mse_", "Same 20 new ICs"),
    ]
    for ax, (prefix, title) in zip(axes, panel_specs):
        for horizon_index, (suffix, label) in enumerate(HORIZONS):
            outcome = f"{prefix}{suffix}"
            estimates, lows, highs = [], [], []
            for train_interval in intervals:
                group = subset_interval(rows, train_interval)
                estimate = parameter_beta(group, outcome)
                low, high = bootstrap_parameter_beta(
                    group,
                    outcome,
                    bootstrap_samples,
                    rng,
                )
                estimates.append(estimate)
                lows.append(low)
                highs.append(high)
            estimates = np.asarray(estimates)
            lows = np.asarray(lows)
            highs = np.asarray(highs)
            yerr = np.vstack(
                [
                    np.maximum(0.0, estimates - lows),
                    np.maximum(0.0, highs - estimates),
                ]
            )
            ax.errorbar(
                intervals,
                estimates,
                yerr=yerr,
                marker=["o", "s", "^"][horizon_index],
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        ax.axhline(0.0, color="0.25", linewidth=0.9)
        ax.set_xticks(intervals, [f"{value:g}" for value in intervals])
        ax.set_xlabel(r"$T_{\mathrm{train}}$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(
        "standardized parameter-error effect\n(controlling VF error and regularization)"
    )
    axes[1].legend(title="fixed horizon", fontsize=8)
    fig.suptitle("Does Parameter Recovery Matter More at Short Training Intervals?")
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_parameter_effect_vs_interval.png", dpi=220)
    plt.close(fig)


def residual_rank_coordinates(rows, outcome):
    parameter, vf, y, mask = outcome_arrays(rows, outcome)
    profiles = profile_values(rows, mask)
    controls = np.column_stack([rankdata(vf), regularization_dummies(profiles)])
    x = residualize(rankdata(parameter), controls)
    y_residual = residualize(rankdata(y), controls)
    selected = [row for row, keep in zip(rows, mask) if keep]
    return x, y_residual, selected


def plot_residual_relationship(rows, output_dir):
    intervals = sorted_intervals(rows)
    fig, axes = plt.subplots(
        1,
        len(intervals),
        figsize=(4.0 * len(intervals), 4.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    outcome = "candidate_ic_extrap_mse_h5"
    for ax, train_interval in zip(axes[0], intervals):
        group = subset_interval(rows, train_interval)
        x, y, selected = residual_rank_coordinates(group, outcome)
        for profile in REG_ORDER:
            mask = np.asarray(
                [row.get("regularization_profile", "none") == profile for row in selected]
            )
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=REG_COLORS[profile],
                    marker=REG_MARKERS[profile],
                    s=34,
                    alpha=0.78,
                    label=REG_LABELS[profile],
                )
        if x.size >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(line_x, intercept + slope * line_x, color="0.2", linewidth=1.4)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8)
        ax.set_title(rf"$T_{{train}}={train_interval:g}$")
        ax.set_xlabel("parameter-error rank residual")
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("H=5 new-IC error rank residual")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=4,
            fontsize=8,
        )
    fig.suptitle("Parameter–Extrapolation Relationship After Removing VF and Reg Effects")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_dir / "fig02_partial_relationship_by_interval.png", dpi=220)
    plt.close(fig)


def plot_interaction_forest(interaction_rows, output_dir):
    labels = {
        "same_ic_extrap_mse_h2p5": "same IC, H=2.5",
        "same_ic_extrap_mse_h5": "same IC, H=5",
        "same_ic_extrap_mse_h10": "same IC, H=10",
        "candidate_ic_extrap_mse_h2p5": "new IC, H=2.5",
        "candidate_ic_extrap_mse_h5": "new IC, H=5",
        "candidate_ic_extrap_mse_h10": "new IC, H=10",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y_positions = np.arange(len(interaction_rows))[::-1]
    for y_position, row in zip(y_positions, interaction_rows):
        estimate = parse_float(row["parameter_error_x_train_interval_interaction"])
        low = parse_float(row["bootstrap_ci_low"])
        high = parse_float(row["bootstrap_ci_high"])
        is_candidate = row["outcome"].startswith("candidate")
        if not all(np.isfinite(value) for value in (estimate, low, high)):
            continue
        ax.errorbar(
            estimate,
            y_position,
            xerr=[
                [max(0.0, estimate - low)],
                [max(0.0, high - estimate)],
            ],
            marker="s" if is_candidate else "o",
            color="C1" if is_candidate else "C0",
            capsize=4,
            markersize=7,
        )
    ax.axvline(0.0, color="0.25", linewidth=1.0)
    ax.set_yticks(
        y_positions,
        [labels.get(row["outcome"], row["outcome"]) for row in interaction_rows],
    )
    ax.set_xlabel(r"interaction: parameter error $\times\ T_{train}$")
    ax.set_title("Primary Test: Negative Values Support Stronger Short-Interval Importance")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_interaction_forest.png", dpi=220)
    plt.close(fig)


def oracle_gain_values(rows, train_interval, field):
    values = np.asarray(
        [parse_float(row.get(field)) for row in subset_interval(rows, train_interval)]
    )
    return values[np.isfinite(values) & (values > 0)]


def plot_oracle_gain(rows, output_dir):
    intervals = sorted_intervals(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    panel_specs = [
        ("oracle_gain_same_ic_", "Same IC"),
        ("oracle_gain_candidate_ic_", "Same 20 new ICs"),
    ]
    plotted_bounds = []
    for ax, (prefix, title) in zip(axes, panel_specs):
        for horizon_index, (suffix, label) in enumerate(HORIZONS):
            medians, lows, highs = [], [], []
            for train_interval in intervals:
                values = oracle_gain_values(rows, train_interval, f"{prefix}{suffix}")
                if values.size:
                    low, median, high = np.percentile(values, [25, 50, 75])
                else:
                    low = median = high = np.nan
                medians.append(median)
                lows.append(low)
                highs.append(high)
                if np.isfinite(low) and np.isfinite(high):
                    plotted_bounds.extend([low, high])
            medians = np.asarray(medians)
            ax.errorbar(
                intervals,
                medians,
                yerr=np.vstack([medians - lows, np.asarray(highs) - medians]),
                marker=["o", "s", "^"][horizon_index],
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        ax.axhline(1.0, color="0.25", linewidth=0.9, linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks(intervals, [f"{value:g}" for value in intervals])
        ax.set_xlabel(r"$T_{\mathrm{train}}$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    if plotted_bounds:
        axes[0].set_ylim(
            max(min(plotted_bounds) / 1.35, 1e-12),
            max(plotted_bounds) * 1.35,
        )
    axes[0].set_ylabel("base MSE / oracle-parameter MSE")
    axes[1].legend(title="fixed horizon", fontsize=8)
    fig.suptitle("Oracle Parameter Replacement Gain (>1 Improves, <1 Worsens)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_oracle_gain_vs_interval.png", dpi=220)
    plt.close(fig)


def heatmap_matrix(rows, field, intervals, profiles):
    matrix = np.full((len(profiles), len(intervals)), np.nan)
    for row_index, profile in enumerate(profiles):
        for column, train_interval in enumerate(intervals):
            values = np.asarray(
                [
                    parse_float(row.get(field))
                    for row in subset_interval(rows, train_interval)
                    if row.get("regularization_profile", "none") == profile
                ]
            )
            values = values[np.isfinite(values) & (values > 0)]
            if values.size:
                matrix[row_index, column] = np.median(values)
    return matrix


def plot_recovery_heatmaps(rows, output_dir):
    intervals = sorted_intervals(rows)
    profiles = [profile for profile in REG_ORDER if any(
        row.get("regularization_profile", "none") == profile for row in rows
    )]
    specs = [
        ("parameter_rel_error", "Parameter relative error"),
        ("model_vf_rel_error", "Global VF relative error"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for ax, (field, title) in zip(axes, specs):
        matrix = heatmap_matrix(rows, field, intervals, profiles)
        image = ax.imshow(np.log10(matrix), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(intervals)), [f"{value:g}" for value in intervals])
        ax.set_yticks(np.arange(len(profiles)), [REG_LABELS[p] for p in profiles])
        ax.set_xlabel(r"$T_{\mathrm{train}}$")
        ax.set_title(title)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(r"$\log_{10}$(relative error)")
    fig.suptitle("Did the Experimental Conditions Create Recovery Variation?")
    fig.tight_layout()
    fig.savefig(output_dir / "fig05_recovery_heatmaps.png", dpi=220)
    plt.close(fig)


def visualize(args):
    script_dir = Path(__file__).resolve().parent
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    input_csv = input_csv if input_csv.is_absolute() else script_dir / input_csv
    output_dir = output_dir if output_dir.is_absolute() else script_dir / output_dir
    main_dir = output_dir / "main"
    tables_dir = output_dir / "tables"
    main_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_rows(input_csv)
    rows = successful_rows(all_rows)
    if not rows:
        raise RuntimeError(f"No successful rows found in {input_csv}")
    validate_single_protocol(rows)

    interval_rows = summarize_by_interval(rows)
    interaction_rows = summarize_interactions(
        rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    write_rows(tables_dir / "parameter_importance_by_interval.csv", interval_rows)
    write_rows(
        tables_dir / "parameter_importance_interaction_test.csv",
        interaction_rows,
    )

    plot_parameter_effect(
        rows,
        main_dir,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    plot_residual_relationship(rows, main_dir)
    plot_interaction_forest(interaction_rows, main_dir)
    plot_oracle_gain(rows, main_dir)
    plot_recovery_heatmaps(rows, main_dir)
    print(f"Saved parameter-importance visualizations to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the carrying parameter-importance sweep."
    )
    parser.add_argument(
        "--input-csv",
        default=(
            "carrying_parameter_importance_sweep_results/"
            "carrying_parameter_importance_sweep_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="carrying_parameter_importance_visualizations",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
