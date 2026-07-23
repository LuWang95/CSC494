import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_INTERVAL_ORDER = [1.25, 2.5, 5.0, 7.5]
TRUE_PHYSICS = np.array([1.0, 0.05, 1.5, 0.03])
PHYSICS_COLUMNS = [
    "learned_f_physics_a",
    "learned_f_physics_b",
    "learned_f_physics_r",
    "learned_f_physics_z",
]
HORIZONS = [
    ("h2p5", 2.5),
    ("h5", 5.0),
    ("h10", 10.0),
]


def parse_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def read_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if int(parse_float(row.get("evaluation_schema_version"), -1)) in {2, 3}
        and row.get("evaluation_status", "ok") in {"", "ok"}
    ]


def load_details(path):
    with path.open("rb") as f:
        return pickle.load(f).get("details", {})


def intervals(rows):
    present = {parse_float(row.get("train_interval")) for row in rows}
    ordered = [value for value in TRAIN_INTERVAL_ORDER if value in present]
    return ordered + sorted(present - set(ordered))


def rows_at(rows, train_interval):
    return [row for row in rows if parse_float(row.get("train_interval")) == train_interval]


def values_at(rows, train_interval, field):
    values = np.asarray(
        [parse_float(row.get(field)) for row in rows_at(rows, train_interval)],
        dtype=float,
    )
    return values[np.isfinite(values)]


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return tuple(np.percentile(values, [50, 25, 75]))


def parameter_error(row):
    learned = np.asarray([parse_float(row.get(name)) for name in PHYSICS_COLUMNS])
    if not np.all(np.isfinite(learned)):
        return np.nan
    return float(np.linalg.norm(learned - TRUE_PHYSICS) / np.linalg.norm(TRUE_PHYSICS))


def detail_for(details, row):
    target_t = parse_float(row.get("train_interval"))
    target_ratio = int(parse_float(row.get("ratio")))
    target_noise = parse_float(row.get("noise_level"), 0.0)
    target_radius = parse_float(row.get("coverage_radius"), 2.5)
    target_seed = int(parse_float(row.get("seed")))
    legacy_key = (target_t, target_ratio, target_noise, target_radius, target_seed)
    current_key = (
        *legacy_key,
        int(parse_float(row.get("chunk_size"), 100)),
        parse_float(row.get("stage_scale"), 1.0),
        int(parse_float(row.get("training_protocol_version"), 2)),
    )
    for key in (current_key, legacy_key):
        if key in details:
            return details[key]
    for candidate_key, detail in details.items():
        if not isinstance(candidate_key, tuple) or len(candidate_key) not in {5, 8}:
            continue
        protocol_matches = len(candidate_key) == 5 or (
            int(candidate_key[5]) == current_key[5]
            and np.isclose(candidate_key[6], current_key[6])
            and int(candidate_key[7]) == current_key[7]
        )
        if (
            protocol_matches
            and np.isclose(candidate_key[0], target_t)
            and int(candidate_key[1]) == target_ratio
            and np.isclose(candidate_key[2], target_noise)
            and np.isclose(candidate_key[3], target_radius)
            and int(candidate_key[4]) == target_seed
        ):
            return detail
    raise KeyError(f"Missing detail record for T={target_t:g}, seed={target_seed}")


def style_axis(ax, ylabel, title, train_intervals):
    ax.set_xlabel(r"$T_{\mathrm{train}}$")
    ax.set_xticks(train_intervals, [f"{value:g}" for value in train_intervals])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def plot_same_ic(rows, output_dir):
    train_intervals = intervals(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True)
    for ax, (suffix, horizon) in zip(axes, HORIZONS):
        field = f"same_ic_extrap_mse_{suffix}"
        seeds = sorted({int(parse_float(row["seed"])) for row in rows})
        for seed in seeds:
            paired = []
            for train_interval in train_intervals:
                matches = [
                    parse_float(row.get(field))
                    for row in rows_at(rows, train_interval)
                    if int(parse_float(row.get("seed"))) == seed
                ]
                paired.append(matches[0] if matches else np.nan)
            ax.plot(train_intervals, paired, color="0.72", linewidth=0.8, alpha=0.65)
        medians, lows, highs = [], [], []
        for train_interval in train_intervals:
            median, low, high = median_iqr(values_at(rows, train_interval, field))
            medians.append(median)
            lows.append(low)
            highs.append(high)
        ax.plot(train_intervals, medians, marker="o", linewidth=2.2, color="C0")
        ax.fill_between(train_intervals, lows, highs, color="C0", alpha=0.18)
        ax.set_yscale("log")
        style_axis(ax, "MSE", rf"Same IC, fixed $H={horizon:g}$", train_intervals)
    fig.suptitle("Training Interval vs Same-IC Extrapolation (paired seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_same_ic_fixed_horizons.png", dpi=220)
    plt.close(fig)


def plot_candidates(rows, output_dir):
    train_intervals = intervals(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True)
    for ax, (suffix, horizon) in zip(axes, HORIZONS):
        field = f"candidate_ic_extrap_mse_{suffix}"
        medians, lows, highs = [], [], []
        for train_interval in train_intervals:
            median, low, high = median_iqr(values_at(rows, train_interval, field))
            medians.append(median)
            lows.append(low)
            highs.append(high)
        ax.plot(train_intervals, medians, marker="o", linewidth=2.2, color="C1")
        ax.fill_between(train_intervals, lows, highs, color="C1", alpha=0.18)
        ax.set_yscale("log")
        style_axis(
            ax,
            "MSE",
            rf"Same 20 new ICs, fixed $H={horizon:g}$",
            train_intervals,
        )
    fig.suptitle("Training Interval vs New-IC Extrapolation (median and seed IQR)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig02_candidate_fixed_horizons.png", dpi=220)
    plt.close(fig)


def candidate_matrices(rows, details, detail_field):
    train_intervals = intervals(rows)
    columns = []
    for train_interval in train_intervals:
        per_seed = [
            np.asarray(detail_for(details, row)[detail_field], dtype=float)
            for row in rows_at(rows, train_interval)
        ]
        columns.append(np.nanmedian(np.stack(per_seed), axis=0))
    return train_intervals, np.stack(columns, axis=1)


def plot_candidate_error_heatmap(rows, details, output_dir):
    train_intervals, matrix = candidate_matrices(rows, details, "candidate_extrap_mse_h5")
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    image = ax.imshow(np.log10(np.maximum(matrix, 1e-16)), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(train_intervals)), [f"{value:g}" for value in train_intervals])
    ax.set_yticks(np.arange(matrix.shape[0]), [str(i) for i in range(matrix.shape[0])])
    ax.set_xlabel(r"$T_{\mathrm{train}}$")
    ax.set_ylabel("candidate IC index")
    ax.set_title("New-IC Extrapolation Error, Fixed H=5 (median across seeds)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}$(MSE)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_candidate_h5_error_heatmap.png", dpi=220)
    plt.close(fig)


def plot_coverage_heatmap(rows, details, output_dir):
    train_intervals, matrix = candidate_matrices(
        rows, details, "candidate_coverage_fraction_h10"
    )
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="cividis")
    ax.set_xticks(np.arange(len(train_intervals)), [f"{value:g}" for value in train_intervals])
    ax.set_yticks(np.arange(matrix.shape[0]), [str(i) for i in range(matrix.shape[0])])
    ax.set_xlabel(r"$T_{\mathrm{train}}$")
    ax.set_ylabel("candidate IC index")
    ax.set_title("Future-State Coverage Fraction over Fixed H=10")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("coverage fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_candidate_coverage_heatmap.png", dpi=220)
    plt.close(fig)


def plot_coverage_vs_error(rows, details, output_dir):
    train_intervals, coverage = candidate_matrices(
        rows, details, "candidate_coverage_fraction_h10"
    )
    _, error = candidate_matrices(rows, details, "candidate_extrap_mse_h5")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for column, train_interval in enumerate(train_intervals):
        ax.scatter(
            coverage[:, column],
            error[:, column],
            s=34,
            alpha=0.75,
            label=rf"$T_{{train}}={train_interval:g}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("future-state coverage fraction (fixed H=10)")
    ax.set_ylabel("new-IC extrapolation MSE (fixed H=5)")
    ax.set_title("Coverage and Extrapolation Error for the Same Candidate ICs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig05_coverage_vs_candidate_error.png", dpi=220)
    plt.close(fig)


def plot_global_recovery(rows, output_dir):
    train_intervals = intervals(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    specs = [
        ("model_vf_rel_error", "global vector-field relative error", "Global VF recovery"),
        ("parameter", "global parameter relative error", "Physics-parameter recovery"),
    ]
    for ax, (field, ylabel, title) in zip(axes, specs):
        medians, lows, highs = [], [], []
        for train_interval in train_intervals:
            group = rows_at(rows, train_interval)
            values = (
                [parameter_error(row) for row in group]
                if field == "parameter"
                else [parse_float(row.get(field)) for row in group]
            )
            median, low, high = median_iqr(values)
            medians.append(median)
            lows.append(low)
            highs.append(high)
        ax.plot(train_intervals, medians, marker="o", linewidth=2.2)
        ax.fill_between(train_intervals, lows, highs, alpha=0.18)
        ax.set_yscale("log")
        style_axis(ax, ylabel, title, train_intervals)
    fig.tight_layout()
    fig.savefig(output_dir / "fig06_global_recovery.png", dpi=220)
    plt.close(fig)


def plot_group_counts(rows, output_dir):
    train_intervals = intervals(rows)
    fields = [
        ("n_covered_ics", "covered", "C2"),
        ("n_partial_ics", "partial", "C1"),
        ("n_uncovered_ics", "uncovered", "C3"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bottoms = np.zeros(len(train_intervals))
    for field, label, color in fields:
        counts = np.asarray(
            [np.median(values_at(rows, value, field)) for value in train_intervals]
        )
        ax.bar(train_intervals, counts, bottom=bottoms, width=0.65, label=label, color=color)
        bottoms += counts
    ax.set_xlabel(r"$T_{\mathrm{train}}$")
    ax.set_ylabel("number of candidate ICs")
    ax.set_title("Descriptive Coverage Groups over the Same Fixed H=10 Window")
    ax.set_ylim(0, 21)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig07_coverage_group_counts.png", dpi=220)
    plt.close(fig)


def write_tables(rows, details, output_dir):
    run_fields = [
        "train_interval",
        "seed",
        "same_ic_extrap_mse_h2p5",
        "same_ic_extrap_mse_h5",
        "same_ic_extrap_mse_h10",
        "candidate_ic_extrap_mse_h2p5",
        "candidate_ic_extrap_mse_h5",
        "candidate_ic_extrap_mse_h10",
        "model_vf_rel_error",
        "parameter_rel_error",
    ]
    with (output_dir / "run_comparable_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        for row in rows:
            record = {field: row.get(field, "") for field in run_fields}
            record["parameter_rel_error"] = parameter_error(row)
            writer.writerow(record)

    candidate_fields = [
        "train_interval", "seed", "candidate_index", "y0_prey", "y0_predator",
        "coverage_fraction_h10", "coverage_group_h10", "extrap_mse_h2p5",
        "extrap_mse_h5", "extrap_mse_h10",
    ]
    with (output_dir / "candidate_comparable_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for row in rows:
            detail = detail_for(details, row)
            for index, y0 in enumerate(np.asarray(detail["candidate_y0"])):
                writer.writerow(
                    {
                        "train_interval": row["train_interval"],
                        "seed": row["seed"],
                        "candidate_index": index,
                        "y0_prey": y0[0],
                        "y0_predator": y0[1],
                        "coverage_fraction_h10": detail["candidate_coverage_fraction_h10"][index],
                        "coverage_group_h10": detail["candidate_coverage_group_h10"][index],
                        "extrap_mse_h2p5": detail["candidate_extrap_mse_h2p5"][index],
                        "extrap_mse_h5": detail["candidate_extrap_mse_h5"][index],
                        "extrap_mse_h10": detail["candidate_extrap_mse_h10"][index],
                    }
                )


def visualize(args):
    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir = input_dir if input_dir.is_absolute() else script_dir / input_dir
    output_dir = output_dir if output_dir.is_absolute() else script_dir / output_dir
    main_dir = output_dir / "main"
    tables_dir = output_dir / "tables"
    main_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_dir / "carrying_train_interval_sweep_summary.csv")
    details = load_details(input_dir / "carrying_train_interval_sweep_details.pkl")
    if not rows:
        raise RuntimeError("No successful interval results found; rerun the experiment first.")
    current_rows = [
        row
        for row in rows
        if int(parse_float(row.get("evaluation_schema_version"), -1)) == 3
    ]
    signatures = {
        (
            row.get("training_protocol_version"),
            row.get("chunk_size"),
            row.get("stage_scale"),
        )
        for row in current_rows
    }
    if len(signatures) > 1:
        raise RuntimeError(
            "Visualization input mixes multiple training protocols; use separate output directories."
        )

    plot_same_ic(rows, main_dir)
    plot_candidates(rows, main_dir)
    plot_candidate_error_heatmap(rows, details, main_dir)
    plot_coverage_heatmap(rows, details, main_dir)
    plot_coverage_vs_error(rows, details, main_dir)
    plot_global_recovery(rows, main_dir)
    plot_group_counts(rows, main_dir)
    write_tables(rows, details, tables_dir)
    print(f"Saved comparable train-interval visualizations to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize comparable fixed-horizon train-interval results."
    )
    parser.add_argument("--input-dir", default="carrying_train_interval_sweep_results")
    parser.add_argument("--output-dir", default="carrying_train_interval_sweep_visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
