import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


METHOD_ORDER = ["forward_euler", "heun", "rk4", "diffrax_tsit5"]
METHOD_LABELS = {
    "forward_euler": "Euler",
    "heun": "Heun",
    "rk4": "RK4",
    "diffrax_tsit5": "Diffrax",
}
METHOD_COLORS = {
    "forward_euler": "tab:blue",
    "heun": "tab:orange",
    "rk4": "tab:green",
    "diffrax_tsit5": "tab:red",
}
REG_ORDER = ["none", "l2_small", "ortho_small", "l2_plus_ortho"]
REG_LABELS = {
    "none": "none",
    "l2_small": "L2",
    "ortho_small": "orthogonality",
    "l2_plus_ortho": "L2 + orthogonality",
}
REG_MARKERS = {
    "none": "o",
    "l2_small": "s",
    "ortho_small": "^",
    "l2_plus_ortho": "D",
}
RATIO_LABELS = ["1", "2", "4", "8", "16", "adaptive"]
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
STATE_SCALE = np.array([50.0, 50.0])
TRUE_A, TRUE_B, TRUE_R, TRUE_Z, TRUE_C = 1.0, 0.05, 1.5, 0.03, 0.001
T_TRAIN = 5.0
T_FINAL = 20.0
REPRESENTATIVE_Y0 = np.array([15.0, 25.0])
VALIDATION_Y0 = np.array(
    [
        [12.0, 35.0],
        [25.0, 12.0],
        [32.0, 28.0],
        [7.0, 35.0],
    ]
)
SUCCESS_THRESHOLDS = [0.5, 1.0, 5.0]
# Matched vector-field band for conditional parameter-vs-extrapolation analysis.
# "center_tolerance" uses |E_VF - center| < tolerance (±10% around 1.5e-3 by default).
# "interval" uses an explicit [low, high) band instead.
VF_MATCHED_MODE = "center_tolerance"
VF_MATCHED_CENTER = 0.0015
VF_MATCHED_REL_TOLERANCE = 0.05  # ±5% around center
VF_MATCHED_TOLERANCE = VF_MATCHED_CENTER * VF_MATCHED_REL_TOLERANCE
VF_MATCHED_INTERVAL = (0.0012, 0.0015)
VF_MATCHED_LEGACY_BAND = (0.001, 0.002)
VF_ERROR_BINS = [
    (0.0012, 0.0015, r"$1.2\times10^{-3}$ to $1.5\times10^{-3}$"),
    (0.0015, 0.0025, r"$1.5\times10^{-3}$ to $2.5\times10^{-3}$"),
    (0.0025, 0.008, r"$2.5\times10^{-3}$ to $8\times10^{-3}$"),
]
PARAM_GOOD_THRESHOLD = 0.08
PARAM_POOR_THRESHOLD = 0.10


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


def method_sort_key(method):
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), ""
    return len(METHOD_ORDER), str(method)


def reg_sort_key(reg):
    if reg in REG_ORDER:
        return REG_ORDER.index(reg), ""
    return len(REG_ORDER), str(reg)


def config_sort_key(config_name):
    if config_name == "diffrax_tsit5":
        return method_sort_key("diffrax_tsit5"), 10**9
    parts = config_name.rsplit("_ratio", 1)
    method = parts[0]
    ratio = parse_int(parts[1]) if len(parts) == 2 else None
    return method_sort_key(method), ratio if ratio is not None else 10**9


def split_config(config_name):
    if config_name == "diffrax_tsit5":
        return "diffrax_tsit5", None
    parts = config_name.rsplit("_ratio", 1)
    if len(parts) != 2:
        return config_name, None
    return parts[0], parse_int(parts[1])


def ratio_label(row):
    method = row.get("training_method", "")
    if method == "diffrax_tsit5":
        return "adaptive"
    ratio = parse_int(row.get("ratio"))
    return str(ratio) if ratio is not None else ""


def ratio_x_position(ratio_label_value):
    return RATIO_LABELS.index(ratio_label_value)


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


def sorted_methods(rows):
    return sorted(
        {row.get("training_method") for row in rows if row.get("training_method")},
        key=method_sort_key,
    )


def sorted_regs(rows):
    return sorted(
        {
            row.get("regularization_profile", "none")
            for row in rows
            if row.get("regularization_profile", "none")
        },
        key=reg_sort_key,
    )


def subset(rows, config_name=None, method=None, ratio_label_value=None, reg=None, seed=None):
    out = rows
    if config_name is not None:
        out = [row for row in out if row.get("training_config") == config_name]
    if method is not None:
        out = [row for row in out if row.get("training_method") == method]
    if ratio_label_value is not None:
        out = [row for row in out if ratio_label(row) == ratio_label_value]
    if reg is not None:
        out = [row for row in out if row.get("regularization_profile", "none") == reg]
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


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    return median, median - q25, q75 - median


def correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size < 2 or np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size < 2:
        return np.nan
    result = spearmanr(x_values, y_values)
    return float(result.correlation)


def log_log_correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
    if np.sum(mask) < 2:
        return np.nan
    return correlation(np.log10(x_values[mask]), np.log10(y_values[mask]))


def parameter_rel_error(row):
    squared_terms = []
    for _, column, true_value in PHYSICS_PARAMS:
        value = parse_float(row.get(column))
        if not np.isfinite(value) or true_value == 0:
            return np.nan
        squared_terms.append(((value - true_value) / true_value) ** 2)
    return float(np.sqrt(np.sum(squared_terms)))


def param_rel_error_column(param_name):
    return f"param_rel_error_{param_name}"


_VF_EXTRAP_FIT = (np.nan, np.nan)


def add_derived_metrics(rows):
    for row in rows:
        row["parameter_rel_error"] = parameter_rel_error(row)
        for param_name, column, true_value in PHYSICS_PARAMS:
            value = parse_float(row.get(column))
            if np.isfinite(value) and true_value != 0:
                row[param_rel_error_column(param_name)] = abs(value - true_value) / abs(true_value)
            else:
                row[param_rel_error_column(param_name)] = np.nan
    return add_extrapolation_residuals(rows)


def vf_error_value(row):
    return parse_float(row.get("model_vf_rel_error"))


def filter_rows_by_vf_band(rows, low, high):
    filtered = []
    for row in rows:
        vf = vf_error_value(row)
        if np.isfinite(vf) and low <= vf < high:
            filtered.append(row)
    return filtered


def resolve_matched_vf_band():
    if VF_MATCHED_MODE == "interval":
        low, high = VF_MATCHED_INTERVAL
        label = rf"${low:.1g} \le E_{{\mathrm{{VF}}}} < {high:.1g}$"
        return low, high, label
    low = VF_MATCHED_CENTER - VF_MATCHED_TOLERANCE
    high = VF_MATCHED_CENTER + VF_MATCHED_TOLERANCE
    label = (
        rf"$|E_{{\mathrm{{VF}}}} - {VF_MATCHED_CENTER:.1g}|"
        rf" < {VF_MATCHED_TOLERANCE:.1g}$"
        rf" (\pm {int(VF_MATCHED_REL_TOLERANCE * 100)}\%)"
    )
    return low, high, label


def filter_rows_by_matched_vf(rows):
    low, high, _ = resolve_matched_vf_band()
    return filter_rows_by_vf_band(rows, low, high)


def fit_log_vf_extrapolation_model(rows):
    vf_values = []
    extrap_values = []
    for row in rows:
        vf = vf_error_value(row)
        extrap = parse_float(row.get("extrapolate_mse"))
        if np.isfinite(vf) and np.isfinite(extrap) and vf > 0 and extrap > 0:
            vf_values.append(vf)
            extrap_values.append(extrap)
    if len(vf_values) < 2:
        return np.nan, np.nan
    log_vf = np.log10(np.asarray(vf_values, dtype=float))
    log_extrap = np.log10(np.asarray(extrap_values, dtype=float))
    design = np.column_stack([np.ones(log_vf.shape), log_vf])
    intercept, slope = np.linalg.lstsq(design, log_extrap, rcond=None)[0]
    return float(intercept), float(slope)


def add_extrapolation_residuals(rows):
    global _VF_EXTRAP_FIT
    intercept, slope = fit_log_vf_extrapolation_model(rows)
    _VF_EXTRAP_FIT = (intercept, slope)
    for row in rows:
        vf = vf_error_value(row)
        extrap = parse_float(row.get("extrapolate_mse"))
        if (
            np.isfinite(intercept)
            and np.isfinite(slope)
            and np.isfinite(vf)
            and np.isfinite(extrap)
            and vf > 0
            and extrap > 0
        ):
            predicted_log = intercept + slope * np.log10(vf)
            row["extrapolate_log_residual"] = float(np.log10(extrap) - predicted_log)
            row["extrapolate_residual_factor"] = float(extrap / (10.0**predicted_log))
        else:
            row["extrapolate_log_residual"] = np.nan
            row["extrapolate_residual_factor"] = np.nan
    return rows


def log_linear_correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0)
    if np.sum(mask) < 2:
        return np.nan
    return correlation(np.log10(x_values[mask]), y_values[mask])


def subset_correlation_lines(rows, x_metric, y_metric, log_x=True, log_y=True):
    x = [parse_float(row.get(x_metric)) for row in rows]
    y = [parse_float(row.get(y_metric)) for row in rows]
    lines = [f"n = {len(rows)}"]
    if log_x and log_y:
        lines.append(f"log-log Pearson r = {log_log_correlation(x, y):.3f}")
    elif log_x and not log_y:
        lines.append(f"log-linear Pearson r = {log_linear_correlation(x, y):.3f}")
    else:
        lines.append(f"Pearson r = {correlation(x, y):.3f}")
    lines.append(f"Spearman rho = {spearman(x, y):.3f}")
    return "\n".join(lines)


def write_conditional_analysis_summary(rows, output_dir):
    intercept, slope = _VF_EXTRAP_FIT
    summary_rows = []

    def append_row(**kwargs):
        summary_rows.append(
            {
                "analysis": kwargs.get("analysis", ""),
                "vf_low": kwargs.get("vf_low", np.nan),
                "vf_high": kwargs.get("vf_high", np.nan),
                "vf_bin_label": kwargs.get("vf_bin_label", ""),
                "n": kwargs.get("n", np.nan),
                "log_log_pearson": kwargs.get("log_log_pearson", np.nan),
                "spearman": kwargs.get("spearman", np.nan),
                "param_threshold_good": kwargs.get("param_threshold_good", np.nan),
                "param_threshold_poor": kwargs.get("param_threshold_poor", np.nan),
                "n_good_param": kwargs.get("n_good_param", np.nan),
                "n_poor_param": kwargs.get("n_poor_param", np.nan),
                "median_extrap_good_param": kwargs.get("median_extrap_good_param", np.nan),
                "median_extrap_poor_param": kwargs.get("median_extrap_poor_param", np.nan),
                "overall_vf_extrap_loglog_r": kwargs.get("overall_vf_extrap_loglog_r", np.nan),
                "vf_extrap_intercept": kwargs.get("vf_extrap_intercept", np.nan),
                "vf_extrap_slope": kwargs.get("vf_extrap_slope", np.nan),
            }
        )

    vf_all = [parse_float(row.get("model_vf_rel_error")) for row in rows]
    extrap_all = [parse_float(row.get("extrapolate_mse")) for row in rows]
    overall_vf_extrap_r = log_log_correlation(vf_all, extrap_all)

    matched_low, matched_high, matched_label = resolve_matched_vf_band()
    matched_rows = filter_rows_by_matched_vf(rows)
    if matched_rows:
        x = [parse_float(row.get("parameter_rel_error")) for row in matched_rows]
        y = [parse_float(row.get("extrapolate_mse")) for row in matched_rows]
        append_row(
            analysis="matched_vf_band",
            vf_low=matched_low,
            vf_high=matched_high,
            vf_bin_label=matched_label,
            n=len(matched_rows),
            log_log_pearson=log_log_correlation(x, y),
            spearman=spearman(x, y),
            overall_vf_extrap_loglog_r=overall_vf_extrap_r,
        )

        good_param = [
            row
            for row in matched_rows
            if parse_float(row.get("parameter_rel_error")) < PARAM_GOOD_THRESHOLD
        ]
        poor_param = [
            row
            for row in matched_rows
            if parse_float(row.get("parameter_rel_error")) > PARAM_POOR_THRESHOLD
        ]
        if good_param and poor_param:
            good_extrap = metric_values(good_param, "extrapolate_mse")
            poor_extrap = metric_values(poor_param, "extrapolate_mse")
            append_row(
                analysis="param_group_comparison_matched_vf",
                vf_low=matched_low,
                vf_high=matched_high,
                vf_bin_label=matched_label,
                param_threshold_good=PARAM_GOOD_THRESHOLD,
                param_threshold_poor=PARAM_POOR_THRESHOLD,
                n_good_param=len(good_param),
                n_poor_param=len(poor_param),
                median_extrap_good_param=float(np.median(good_extrap)),
                median_extrap_poor_param=float(np.median(poor_extrap)),
            )

    legacy_rows = filter_rows_by_vf_band(rows, *VF_MATCHED_LEGACY_BAND)
    if legacy_rows:
        x = [parse_float(row.get("parameter_rel_error")) for row in legacy_rows]
        y = [parse_float(row.get("extrapolate_mse")) for row in legacy_rows]
        append_row(
            analysis="matched_vf_band_legacy",
            vf_low=VF_MATCHED_LEGACY_BAND[0],
            vf_high=VF_MATCHED_LEGACY_BAND[1],
            n=len(legacy_rows),
            log_log_pearson=log_log_correlation(x, y),
            spearman=spearman(x, y),
            overall_vf_extrap_loglog_r=overall_vf_extrap_r,
        )

    for low, high, label in VF_ERROR_BINS:
        band_rows = filter_rows_by_vf_band(rows, low, high)
        if not band_rows:
            continue
        x = [parse_float(row.get("parameter_rel_error")) for row in band_rows]
        y = [parse_float(row.get("extrapolate_mse")) for row in band_rows]
        append_row(
            analysis="vf_bin",
            vf_low=low,
            vf_high=high,
            vf_bin_label=label,
            n=len(band_rows),
            log_log_pearson=log_log_correlation(x, y),
            spearman=spearman(x, y),
        )

    x = [parse_float(row.get("parameter_rel_error")) for row in rows]
    y = [parse_float(row.get("extrapolate_log_residual")) for row in rows]
    append_row(
        analysis="residual_extrapolation",
        n=len(rows),
        log_log_pearson=log_linear_correlation(x, y),
        spearman=spearman(x, y),
        vf_extrap_intercept=intercept,
        vf_extrap_slope=slope,
    )
    write_csv(output_dir / "conditional_analysis_summary.csv", summary_rows)


def run_key(row):
    return (
        row.get("training_config"),
        row.get("regularization_profile", "none"),
        parse_int(row.get("seed")),
    )


def positive_mask(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    return np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def correlation_groups(rows):
    groups = [("all", rows)]
    for method in sorted_methods(rows):
        method_rows = [row for row in rows if row.get("training_method") == method]
        if method_rows:
            groups.append((method, method_rows))
            for ratio_label_value in sorted(
                {ratio_label(row) for row in method_rows},
                key=lambda label: (0, int(label)) if label.isdigit() else (1, label),
            ):
                ratio_rows = [
                    row
                    for row in method_rows
                    if ratio_label(row) == ratio_label_value
                ]
                if ratio_rows:
                    groups.append((f"{method}_ratio{ratio_label_value}", ratio_rows))
    return groups


def write_correlation_summary(rows, output_dir):
    pairs = [
        ("best_loss", "extrapolate_mse"),
        ("validation_mse", "extrapolate_mse"),
        ("parameter_rel_error", "model_vf_rel_error"),
        ("parameter_rel_error", "extrapolate_mse"),
        ("model_vf_rel_error", "extrapolate_mse"),
        ("residual_mse", "extrapolate_mse"),
        ("residual_rel_error", "extrapolate_mse"),
    ]
    out_rows = []
    for group_name, group_rows in correlation_groups(rows):
        for x_metric, y_metric in pairs:
            x = [parse_float(row.get(x_metric)) for row in group_rows]
            y = [parse_float(row.get(y_metric)) for row in group_rows]
            out_rows.append(
                {
                    "group": group_name,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": len(group_rows),
                    "pearson": correlation(x, y),
                    "log_log_pearson": log_log_correlation(x, y),
                    "spearman": spearman(x, y),
                }
            )
    write_csv(output_dir / "correlation_summary.csv", out_rows)


def write_group_summary(rows, output_dir):
    summary_rows = []
    for config_name in sorted_configs(rows):
        for reg in sorted_regs(rows):
            group = subset(rows, config_name=config_name, reg=reg)
            if not group:
                continue
            summary = {
                "training_config": config_name,
                "training_method": group[0].get("training_method", ""),
                "ratio": group[0].get("ratio", ""),
                "regularization_profile": reg,
                "n_runs": len(group),
            }
            for metric in METRICS + ["parameter_rel_error"]:
                stats = summarize(metric_values(group, metric))
                for stat_name, value in stats.items():
                    summary[f"{metric}_{stat_name}"] = value
            summary_rows.append(summary)
    write_csv(output_dir / "group_summary.csv", summary_rows)


def correlation_annotation(rows, x_metric, y_metric, include_within_solver=True):
    x_all = [parse_float(row.get(x_metric)) for row in rows]
    y_all = [parse_float(row.get(y_metric)) for row in rows]
    lines = [
        f"overall log-log Pearson r = {log_log_correlation(x_all, y_all):.3f}",
        f"overall Spearman rho = {spearman(x_all, y_all):.3f}",
    ]
    if include_within_solver:
        for method in sorted_methods(rows):
            group = [row for row in rows if row.get("training_method") == method]
            x = [parse_float(row.get(x_metric)) for row in group]
            y = [parse_float(row.get(y_metric)) for row in group]
            lines.append(
                f"{METHOD_LABELS.get(method, method)} (pools ratios): "
                f"r={log_log_correlation(x, y):.3f}, rho={spearman(x, y):.3f}"
            )
    return "\n".join(lines)


def plot_solver_ratio_metric(
    rows,
    output_path,
    metric,
    ylabel,
    title,
    log_y=True,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x_positions = np.arange(len(RATIO_LABELS))

    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        summary_x = []
        summary_median = []
        summary_low = []
        summary_high = []

        for ratio_label_value in RATIO_LABELS:
            if method == "diffrax_tsit5":
                if ratio_label_value != "adaptive":
                    continue
                group = subset(rows, method=method)
            else:
                if ratio_label_value == "adaptive":
                    continue
                group = subset(rows, method=method, ratio_label_value=ratio_label_value)

            values = metric_values(group, metric)
            if values.size == 0:
                continue

            x_pos = ratio_x_position(ratio_label_value)
            jitter = np.linspace(-0.14, 0.14, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, x_pos) + jitter,
                values,
                color=color,
                alpha=0.22,
                s=16,
                linewidths=0,
            )

            median, low, high = median_iqr(values)
            summary_x.append(x_pos)
            summary_median.append(median)
            summary_low.append(low)
            summary_high.append(high)

        if summary_x:
            ax.errorbar(
                summary_x,
                summary_median,
                yerr=[summary_low, summary_high],
                color=color,
                marker="o",
                markersize=7,
                linewidth=2.0,
                capsize=4,
                label=label,
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(RATIO_LABELS)
    ax.set_xlabel("training ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="training solver", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_mechanism_scatter(
    rows,
    output_path,
    x_metric,
    y_metric,
    xlabel,
    ylabel,
    title,
    color_by_solver=True,
    marker_by_reg=False,
):
    fig, ax = plt.subplots(figsize=(7.8, 6.0))

    if color_by_solver and marker_by_reg:
        for method in sorted_methods(rows):
            for reg in sorted_regs(rows):
                group = [
                    row
                    for row in rows
                    if row.get("training_method") == method
                    and row.get("regularization_profile", "none") == reg
                ]
                x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
                y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
                mask = positive_mask(x, y)
                if not np.any(mask):
                    continue
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=METHOD_COLORS.get(method, "gray"),
                    marker=REG_MARKERS.get(reg, "o"),
                    alpha=0.78,
                    s=34,
                    edgecolors="white",
                    linewidths=0.3,
                )
        for method in sorted_methods(rows):
            group = [row for row in rows if row.get("training_method") == method]
            x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
            y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
            mask = positive_mask(x, y)
            if np.any(mask):
                ax.scatter(
                    [],
                    [],
                    color=METHOD_COLORS.get(method, "gray"),
                    marker="o",
                    s=40,
                    label=METHOD_LABELS.get(method, method),
                )
        if marker_by_reg:
            for reg in sorted_regs(rows):
                ax.scatter(
                    [],
                    [],
                    color="black",
                    marker=REG_MARKERS.get(reg, "o"),
                    s=40,
                    label=REG_LABELS.get(reg, reg),
                )
    else:
        for method in sorted_methods(rows):
            group = [row for row in rows if row.get("training_method") == method]
            x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
            y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
            mask = positive_mask(x, y)
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=METHOD_COLORS.get(method, "gray"),
                    alpha=0.78,
                    s=36,
                    label=METHOD_LABELS.get(method, method),
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(
        0.03,
        0.97,
        correlation_annotation(rows, x_metric, y_metric),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_within_solver_panels(
    rows,
    output_path,
    x_metric,
    y_metric,
    xlabel,
    ylabel,
    suptitle,
):
    methods = [method for method in METHOD_ORDER if any(row.get("training_method") == method for row in rows)]
    if not methods:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.0 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        group = [row for row in rows if row.get("training_method") == method]
        x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
        y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
        mask = positive_mask(x, y)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], color=METHOD_COLORS.get(method, "gray"), alpha=0.8, s=34)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n(pools all ratios)")
        ax.grid(True, which="both", alpha=0.3)
        ax.text(
            0.03,
            0.97,
            (
                f"log-log r={log_log_correlation(x, y):.3f}\n"
                f"rho={spearman(x, y):.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_parameter_recovery_panels(rows, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), squeeze=False)
    x_positions = np.arange(len(RATIO_LABELS))

    for ax, (param_name, column, true_value) in zip(axes.ravel(), PHYSICS_PARAMS):
        for method in METHOD_ORDER:
            summary_x = []
            summary_median = []
            summary_low = []
            summary_high = []

            for ratio_label_value in RATIO_LABELS:
                if method == "diffrax_tsit5":
                    if ratio_label_value != "adaptive":
                        continue
                    base_group = subset(rows, method=method)
                else:
                    if ratio_label_value == "adaptive":
                        continue
                    base_group = subset(rows, method=method, ratio_label_value=ratio_label_value)

                values = metric_values(base_group, column)
                if values.size == 0:
                    continue

                x_pos = ratio_x_position(ratio_label_value)
                for reg in sorted_regs(rows):
                    group = [
                        row
                        for row in base_group
                        if row.get("regularization_profile", "none") == reg
                    ]
                    reg_values = metric_values(group, column)
                    if reg_values.size == 0:
                        continue
                    jitter = (
                        np.linspace(-0.08, 0.08, reg_values.size)
                        if reg_values.size > 1
                        else np.array([0.0])
                    )
                    ax.scatter(
                        np.full(reg_values.size, x_pos) + jitter,
                        reg_values,
                        color=METHOD_COLORS[method],
                        marker=REG_MARKERS.get(reg, "o"),
                        alpha=0.28,
                        s=18,
                        linewidths=0,
                    )

                median, low, high = median_iqr(values)
                summary_x.append(x_pos)
                summary_median.append(median)
                summary_low.append(low)
                summary_high.append(high)

            if summary_x:
                ax.errorbar(
                    summary_x,
                    summary_median,
                    yerr=[summary_low, summary_high],
                    color=METHOD_COLORS[method],
                    marker="o",
                    capsize=3,
                    linewidth=1.6,
                    label=METHOD_LABELS[method],
                )

        ax.axhline(true_value, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(RATIO_LABELS)
        ax.set_xlabel("training ratio")
        ax.set_ylabel(f"learned {param_name}")
        ax.set_title(param_name)
        ax.grid(True, alpha=0.3)

    for method in METHOD_ORDER:
        axes[0, 0].scatter(
            [],
            [],
            color=METHOD_COLORS[method],
            marker="o",
            s=30,
            label=METHOD_LABELS[method],
        )
    for reg in REG_ORDER:
        axes[0, 1].scatter(
            [],
            [],
            color="gray",
            marker=REG_MARKERS[reg],
            s=30,
            label=REG_LABELS[reg],
        )

    axes[0, 0].legend(title="solver", fontsize=8, loc="best")
    axes[0, 1].legend(title="regularization", fontsize=8, loc="best")
    fig.suptitle("Physics Parameter Recovery Across Solver and Step-Size Configurations", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def success_rate_matrix(rows, threshold):
    methods = [method for method in METHOD_ORDER if any(row.get("training_method") == method for row in rows)]
    ratio_cols = [label for label in RATIO_LABELS if label != "adaptive" or "diffrax_tsit5" in methods]
    matrix = np.full((len(methods), len(ratio_cols)), np.nan)

    for i, method in enumerate(methods):
        for j, ratio_label_value in enumerate(ratio_cols):
            if method == "diffrax_tsit5" and ratio_label_value != "adaptive":
                continue
            if method != "diffrax_tsit5" and ratio_label_value == "adaptive":
                continue
            group = subset(rows, method=method, ratio_label_value=ratio_label_value)
            values = metric_values(group, "extrapolate_mse")
            if values.size:
                matrix[i, j] = float(np.mean(values < threshold))
    return methods, ratio_cols, matrix


def plot_success_rate_heatmaps(rows, output_path):
    fig, axes = plt.subplots(1, len(SUCCESS_THRESHOLDS), figsize=(4.8 * len(SUCCESS_THRESHOLDS), 4.8))
    if len(SUCCESS_THRESHOLDS) == 1:
        axes = [axes]

    im = None
    for ax, threshold in zip(axes, SUCCESS_THRESHOLDS):
        methods, ratio_cols, matrix = success_rate_matrix(rows, threshold)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(ratio_cols)))
        ax.set_xticklabels(ratio_cols)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods])
        ax.set_xlabel("training ratio")
        ax.set_title(f"success rate (< {threshold:g})")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] < 0.55 else "black",
                        fontsize=8,
                    )

    fig.suptitle("Extrapolation Success Rate Across Training Configurations", fontsize=13)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="fraction below threshold")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_extrapolation_heatmap(rows, output_path):
    methods = [method for method in METHOD_ORDER if any(row.get("training_method") == method for row in rows)]
    ratio_cols = RATIO_LABELS
    matrix = np.full((len(methods), len(ratio_cols)), np.nan)

    for i, method in enumerate(methods):
        for j, ratio_label_value in enumerate(ratio_cols):
            if method == "diffrax_tsit5" and ratio_label_value != "adaptive":
                continue
            if method != "diffrax_tsit5" and ratio_label_value == "adaptive":
                continue
            group = subset(rows, method=method, ratio_label_value=ratio_label_value)
            values = metric_values(group, "extrapolate_mse")
            if values.size:
                positive = values[values > 0]
                if positive.size:
                    matrix[i, j] = float(np.median(positive))

    log_matrix = np.full_like(matrix, np.nan)
    mask = np.isfinite(matrix) & (matrix > 0)
    log_matrix[mask] = np.log10(matrix[mask])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(log_matrix, aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(ratio_cols)))
    ax.set_xticklabels(ratio_cols)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods])
    ax.set_xlabel("training ratio")
    ax.set_ylabel("training solver")
    ax.set_title("Median Extrapolation Error by Solver and Training Step Size")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2e}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ median extrapolation MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_regularization_summary(rows, output_path):
    metrics = [
        ("best_loss", "best validation loss"),
        ("model_vf_rel_error", "model vector-field relative error"),
        ("extrapolate_mse", "extrapolation MSE"),
    ]
    regs = [reg for reg in REG_ORDER if any(row.get("regularization_profile", "none") == reg for row in rows)]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        summary_x = []
        summary_median = []
        summary_low = []
        summary_high = []

        for reg in regs:
            values = metric_values(subset(rows, reg=reg), metric)
            if values.size == 0:
                continue
            x_pos = regs.index(reg)
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, x_pos) + jitter,
                values,
                color="gray",
                alpha=0.35,
                s=20,
                linewidths=0,
            )
            median, low, high = median_iqr(values)
            summary_x.append(x_pos)
            summary_median.append(median)
            summary_low.append(low)
            summary_high.append(high)

        if summary_x:
            ax.errorbar(
                summary_x,
                summary_median,
                yerr=[summary_low, summary_high],
                color="black",
                marker="o",
                capsize=4,
                linewidth=1.8,
            )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", alpha=0.3)

    axes[0].set_xticks(np.arange(len(regs)))
    axes[0].set_xticklabels([REG_LABELS[reg] for reg in regs], rotation=20, ha="right")
    axes[0].set_xlabel("regularization profile")
    fig.suptitle(
        "Limited Effect of the Tested Regularization Profiles\n"
        "(pooled across solver, ratio, and seed)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_regularization_solver_interaction(rows, output_path):
    methods = [method for method in METHOD_ORDER if any(row.get("training_method") == method for row in rows)]
    regs = [reg for reg in REG_ORDER if any(row.get("regularization_profile", "none") == reg for row in rows)]
    metrics = [
        ("model_vf_rel_error", "vector-field relative error"),
        ("extrapolate_mse", "extrapolation MSE"),
    ]

    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(3.8 * len(methods), 4.8 * len(metrics)), squeeze=False)
    for row_axes, (metric, metric_label) in zip(axes, metrics):
        for ax, method in zip(row_axes, methods):
            xs = []
            medians = []
            lows = []
            highs = []
            for reg in regs:
                values = metric_values(
                    [
                        row
                        for row in rows
                        if row.get("training_method") == method
                        and row.get("regularization_profile", "none") == reg
                    ],
                    metric,
                )
                if values.size == 0:
                    continue
                x_pos = regs.index(reg)
                jitter = np.linspace(-0.1, 0.1, values.size) if values.size > 1 else np.array([0.0])
                ax.scatter(
                    np.full(values.size, x_pos) + jitter,
                    values,
                    color="gray",
                    alpha=0.35,
                    s=16,
                    linewidths=0,
                )
                median, low, high = median_iqr(values)
                xs.append(x_pos)
                medians.append(median)
                lows.append(low)
                highs.append(high)
            if xs:
                ax.errorbar(xs, medians, yerr=[lows, highs], color="black", marker="o", capsize=3)
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(regs)))
            ax.set_xticklabels([REG_LABELS[reg] for reg in regs], rotation=25, ha="right", fontsize=7)
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.grid(True, which="both", axis="y", alpha=0.3)
            if ax is row_axes[0]:
                ax.set_ylabel(metric_label)

    fig.suptitle("Regularization x Solver Interaction", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def select_representative_runs(rows, trained_models):
    candidates = []
    for row in rows:
        key = run_key(row)
        if key not in trained_models:
            continue
        best_loss = parse_float(row.get("best_loss"))
        extrap = parse_float(row.get("extrapolate_mse"))
        if np.isfinite(best_loss) and np.isfinite(extrap):
            candidates.append((best_loss, extrap, key, row))

    if len(candidates) < 3:
        return []

    losses = np.array([item[0] for item in candidates])
    q25, q75 = np.percentile(losses, [25, 75])
    filtered = [item for item in candidates if q25 <= item[0] <= q75]
    if len(filtered) < 3:
        median_loss = float(np.median(losses))
        tolerance = max(0.25 * median_loss, np.percentile(losses, 75) - np.percentile(losses, 25))
        filtered = [item for item in candidates if abs(item[0] - median_loss) <= tolerance]
    if len(filtered) < 3:
        filtered = candidates

    filtered.sort(key=lambda item: item[1])
    picks = [
        ("good extrapolation", filtered[0]),
        ("intermediate extrapolation", filtered[len(filtered) // 2]),
        ("catastrophic extrapolation", filtered[-1]),
    ]
    return picks


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
            TRUE_A * prey - TRUE_B * prey * predator - TRUE_C * prey * prey,
            -TRUE_R * predator + TRUE_Z * prey * predator,
        ],
        axis=1,
    )


def model_rhs_numpy(states, params):
    residual_scale = float(np.asarray(params.get("residual_scale", 1.0)))
    return (
        f_physics_numpy(states, params["f_physics"])
        + residual_scale * nn_numpy(states, params["nn_params"])
    )


_ROLLOUT_BACKEND = None


def _resolve_rollout_backend():
    global _ROLLOUT_BACKEND
    if _ROLLOUT_BACKEND is not None:
        return _ROLLOUT_BACKEND
    try:
        import jax.numpy as jnp
        from carrying_solver_sweep import solve_model_diffrax, solve_reference

        _ROLLOUT_BACKEND = ("diffrax", jnp, solve_reference, solve_model_diffrax)
        print("[info] Trajectory rollouts use Diffrax/Tsit5 reference solver.")
    except Exception as exc:
        print(f"[warn] Diffrax rollout backend unavailable ({exc}); falling back to RK4.")
        _ROLLOUT_BACKEND = ("rk4", None, None, None)
    return _ROLLOUT_BACKEND


def rk4_step(y, h, rhs):
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_numpy(y0, t_grid, rhs):
    y = np.asarray(y0, dtype=float)
    ys = [y.copy()]
    for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
        y = rk4_step(y, float(t1 - t0), rhs)
        ys.append(y.copy())
    return np.asarray(ys)


def true_trajectory(y0, t_grid):
    backend, jnp, solve_reference, _ = _resolve_rollout_backend()
    if backend == "diffrax":
        ys = solve_reference(jnp.asarray(y0, dtype=float), jnp.asarray(t_grid, dtype=float))
        return np.asarray(ys)
    return rollout_numpy(
        y0,
        t_grid,
        lambda y: true_rhs_numpy(y[None, :])[0],
    )


def model_trajectory(y0, t_grid, trained):
    backend, jnp, _, solve_model_diffrax = _resolve_rollout_backend()
    if backend == "diffrax":
        ys = solve_model_diffrax(
            jnp.asarray(y0, dtype=float),
            trained["params"],
            jnp.asarray(t_grid, dtype=float),
            float(t_grid[-1]),
        )
        return np.asarray(ys)
    params = trained["params"]
    return rollout_numpy(
        y0,
        t_grid,
        lambda y: model_rhs_numpy(y[None, :], params)[0],
    )


def shade_intervals(ax):
    ax.axvspan(0.0, T_TRAIN, color="tab:blue", alpha=0.08, label="training interval")
    ax.axvspan(T_TRAIN, T_FINAL, color="tab:orange", alpha=0.08, label="extrapolation interval")
    ax.axvline(T_TRAIN, color="tab:red", linestyle=":", linewidth=1.1)


def plot_representative_trajectories(rows, trained_models, output_path):
    selected = select_representative_runs(rows, trained_models)
    if not selected:
        return

    t_grid = np.linspace(0.0, T_FINAL, 401)
    true_traj = true_trajectory(REPRESENTATIVE_Y0, t_grid)
    species = ["prey x(t)", "predator y(t)"]

    fig, axes = plt.subplots(len(selected), 2, figsize=(11.5, 3.4 * len(selected)), sharex=True, squeeze=False)
    for row_idx, (label, (_, extrap, key, row)) in enumerate(selected):
        pred_traj = model_trajectory(REPRESENTATIVE_Y0, t_grid, trained_models[key])
        config_name, reg, seed = key
        best_loss = parse_float(row.get("best_loss"))
        for state_idx, species_name in enumerate(species):
            ax = axes[row_idx, state_idx]
            ax.plot(t_grid, true_traj[:, state_idx], color="black", linewidth=1.8, label=reference_trajectory_label())
            ax.plot(
                t_grid,
                pred_traj[:, state_idx],
                linestyle="--",
                color=METHOD_COLORS.get(row.get("training_method"), "tab:blue"),
                linewidth=1.8,
                label="prediction",
            )
            shade_intervals(ax)
            ax.set_ylabel(species_name)
            ax.set_title(
                f"{label}\n"
                f"{METHOD_LABELS.get(row.get('training_method'), row.get('training_method'))}, "
                f"ratio={ratio_label(row)}, {REG_LABELS.get(reg, reg)}, seed={seed}\n"
                f"best loss={best_loss:.3g}, extrapolation MSE={extrap:.3g}"
            )
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and state_idx == 1:
                ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("time t")
    fig.suptitle("Representative Long-Term Rollouts", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_representative_phase_portraits(rows, trained_models, output_path):
    selected = select_representative_runs(rows, trained_models)
    if not selected:
        return

    t_grid = np.linspace(0.0, T_FINAL, 401)
    train_idx = int(np.argmin(np.abs(t_grid - T_TRAIN)))

    fig, axes = plt.subplots(1, len(selected), figsize=(5.4 * len(selected), 4.8), squeeze=False)
    for ax, (label, (_, extrap, key, row)) in zip(axes[0], selected):
        true_main = true_trajectory(REPRESENTATIVE_Y0, t_grid)
        pred_main = model_trajectory(REPRESENTATIVE_Y0, t_grid, trained_models[key])
        config_name, reg, seed = key

        ax.plot(true_main[:, 0], true_main[:, 1], color="black", linewidth=1.8, label=reference_trajectory_label())
        ax.plot(
            pred_main[:, 0],
            pred_main[:, 1],
            linestyle="--",
            color=METHOD_COLORS.get(row.get("training_method"), "tab:blue"),
            linewidth=1.8,
            label="prediction",
        )
        ax.scatter(true_main[0, 0], true_main[0, 1], color="tab:green", s=28, zorder=4)
        ax.scatter(
            true_main[train_idx, 0],
            true_main[train_idx, 1],
            color="tab:red",
            s=36,
            marker="X",
            zorder=4,
            label="training end",
        )

        for y0 in VALIDATION_Y0:
            true_val = true_trajectory(y0, t_grid)
            pred_val = model_trajectory(y0, t_grid, trained_models[key])
            ax.plot(
                true_val[:, 0],
                true_val[:, 1],
                color="black",
                alpha=0.18,
                linewidth=1.0,
            )
            ax.plot(
                pred_val[:, 0],
                pred_val[:, 1],
                linestyle="--",
                color=METHOD_COLORS.get(row.get("training_method"), "tab:blue"),
                alpha=0.18,
                linewidth=1.0,
            )

        ax.set_xlabel("prey x")
        ax.set_ylabel("predator y")
        ax.set_title(
            f"{label}\n"
            f"{METHOD_LABELS.get(row.get('training_method'), row.get('training_method'))}, "
            f"ratio={ratio_label(row)}, extrap MSE={extrap:.3g}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Phase-Space Behavior of Representative Learned Models", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compact_config_label(config_name):
    method, ratio = split_config(config_name)
    if method == "diffrax_tsit5":
        return "Diffrax"
    short = {
        "forward_euler": "Euler",
        "heun": "Heun",
        "rk4": "RK4",
    }.get(method, method)
    return f"{short} r{ratio}"


def seed_pooled_metric(rows, config_name, seed, metric):
    values = metric_values(subset(rows, config_name=config_name, seed=seed), metric)
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def seed_cv_per_config(rows, config_name, metric):
    seeds = sorted(
        {
            parse_int(row.get("seed"))
            for row in rows
            if parse_int(row.get("seed")) is not None
        }
    )
    seed_medians = []
    for seed in seeds:
        value = seed_pooled_metric(rows, config_name, seed, metric)
        if np.isfinite(value):
            seed_medians.append(value)
    if len(seed_medians) < 2:
        return np.nan
    seed_medians = np.asarray(seed_medians, dtype=float)
    mean = float(np.mean(seed_medians))
    if mean <= 0:
        return np.nan
    return float(np.std(seed_medians, ddof=1) / mean)


def plot_seed_stability(rows, output_path):
    metrics = [
        ("extrapolate_mse", "extrapolation MSE"),
        ("model_vf_rel_error", "vector-field relative error"),
        ("parameter_rel_error", "relative parameter error"),
    ]
    configs = sorted_configs(rows)
    config_labels = [compact_config_label(config_name) for config_name in configs]
    x_positions = np.arange(len(configs))
    seeds = sorted({parse_int(row.get("seed")) for row in rows if parse_int(row.get("seed")) is not None})
    seed_colors = plt.cm.tab10(np.linspace(0, 1, max(len(seeds), 1)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for seed_idx, seed in enumerate(seeds):
            ys = []
            valid_x = []
            for x_pos, config_name in enumerate(configs):
                value = seed_pooled_metric(rows, config_name, seed, metric)
                if np.isfinite(value):
                    valid_x.append(x_pos)
                    ys.append(value)
            if ys:
                ax.plot(
                    valid_x,
                    ys,
                    color=seed_colors[seed_idx],
                    marker="o",
                    linewidth=1.2,
                    alpha=0.85,
                    label=f"seed {seed}",
                )

        cv_values = []
        for config_name in configs:
            cv_values.append(seed_cv_per_config(rows, config_name, metric))
        ax2 = ax.twinx()
        ax2.plot(
            x_positions,
            cv_values,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label="CV across seed medians",
        )
        ax2.set_ylabel("coefficient of variation", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(config_labels, rotation=60, ha="right", fontsize=7)
    axes[0].set_xlabel("training configuration (median across regularization)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Seed Stability Across Training Configurations", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pick_extrapolation_median_run(group_rows):
    candidates = []
    for row in group_rows:
        extrap = parse_float(row.get("extrapolate_mse"))
        if np.isfinite(extrap):
            candidates.append((extrap, row))
    if not candidates:
        return None
    target = float(np.median([value for value, _ in candidates]))
    return min(candidates, key=lambda item: abs(item[0] - target))[1]


def plot_error_over_time(details, rows, output_path, t_final=T_FINAL):
    representative_configs = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row.get("training_method") == method]
        if not method_rows:
            continue
        if method == "diffrax_tsit5":
            pick = pick_extrapolation_median_run(method_rows)
            if pick is not None:
                representative_configs.append(pick)
            continue

        by_ratio = {}
        for row in method_rows:
            ratio = ratio_label(row)
            by_ratio.setdefault(ratio, []).append(row)
        for ratio in sorted(by_ratio, key=lambda label: int(label) if label.isdigit() else 99):
            pick = pick_extrapolation_median_run(by_ratio[ratio])
            if pick is not None:
                representative_configs.append(pick)

    if not details:
        return

    first_curve = None
    for row in representative_configs:
        key = run_key(row)
        curves = details.get(key)
        if curves and "extrapolate_rel_error_by_time" in curves:
            first_curve = curves["extrapolate_rel_error_by_time"]
            break
    if first_curve is None:
        return

    n_times = len(first_curve)
    time_grid = np.linspace(0.0, t_final, n_times)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for row in representative_configs:
        key = run_key(row)
        curves = details.get(key)
        if curves is None:
            continue
        error_curve = curves.get("extrapolate_rel_error_by_time")
        if error_curve is None:
            continue
        label = (
            f"{METHOD_LABELS.get(row.get('training_method'), row.get('training_method'))}, "
            f"ratio={ratio_label(row)}, {REG_LABELS.get(row.get('regularization_profile', 'none'), 'none')}"
        )
        ax.semilogy(time_grid, error_curve, linewidth=1.6, alpha=0.9, label=label)

    ax.axvline(T_TRAIN, color="tab:red", linestyle=":", linewidth=1.2, label="extrapolation start")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$\|y_{\mathrm{pred}}(t)-y_{\mathrm{true}}(t)\| / \|y_{\mathrm{true}}(t)\|$")
    ax.set_title("Prediction Error vs Time for Median-Representative Runs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def reference_trajectory_label():
    backend, *_ = _resolve_rollout_backend()
    if backend == "diffrax":
        return "reference (Diffrax/Tsit5)"
    return "reference (RK4)"


def plot_fixed_ratio_correlation_heatmaps(rows, output_path):
    key_pairs = [
        ("model_vf_rel_error", "extrapolate_mse", "Vector-Field vs Extrapolation"),
        ("parameter_rel_error", "model_vf_rel_error", "Parameter vs Vector-Field"),
        ("parameter_rel_error", "extrapolate_mse", "Parameter vs Extrapolation"),
    ]
    methods = [
        method
        for method in METHOD_ORDER
        if any(row.get("training_method") == method for row in rows)
    ]
    if not methods:
        return

    fig, axes = plt.subplots(1, len(key_pairs), figsize=(5.5 * len(key_pairs), 4.8))
    if len(key_pairs) == 1:
        axes = [axes]

    last_im = None
    for ax, (x_metric, y_metric, title) in zip(axes, key_pairs):
        matrix_r = np.full((len(methods), len(RATIO_LABELS)), np.nan)
        for i, method in enumerate(methods):
            for j, ratio_label_value in enumerate(RATIO_LABELS):
                if method == "diffrax_tsit5" and ratio_label_value != "adaptive":
                    continue
                if method != "diffrax_tsit5" and ratio_label_value == "adaptive":
                    continue
                group = subset(rows, method=method, ratio_label_value=ratio_label_value)
                x = [parse_float(row.get(x_metric)) for row in group]
                y = [parse_float(row.get(y_metric)) for row in group]
                matrix_r[i, j] = log_log_correlation(x, y)

        last_im = ax.imshow(matrix_r, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(len(RATIO_LABELS)))
        ax.set_xticklabels(RATIO_LABELS)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods])
        ax.set_xlabel("training ratio")
        ax.set_title(title)
        for i in range(matrix_r.shape[0]):
            for j in range(matrix_r.shape[1]):
                if np.isfinite(matrix_r[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix_r[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(matrix_r[i, j]) > 0.55 else "black",
                    )

    fig.suptitle("Within-Solver Fixed-Ratio Correlations", fontsize=13)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.68])
    fig.colorbar(last_im, cax=cbar_ax, label="log-log Pearson r")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_per_parameter_extrapolation(rows, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), squeeze=False)
    for ax, (param_name, _, _) in zip(axes.ravel(), PHYSICS_PARAMS):
        metric = param_rel_error_column(param_name)
        all_x = [parse_float(row.get(metric)) for row in rows]
        all_y = [parse_float(row.get("extrapolate_mse")) for row in rows]
        for method in sorted_methods(rows):
            group = [row for row in rows if row.get("training_method") == method]
            x = np.array([parse_float(row.get(metric)) for row in group], dtype=float)
            y = np.array([parse_float(row.get("extrapolate_mse")) for row in group], dtype=float)
            mask = positive_mask(x, y)
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=METHOD_COLORS.get(method, "gray"),
                    alpha=0.78,
                    s=32,
                    label=METHOD_LABELS.get(method, method),
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"relative error in {param_name}")
        ax.set_ylabel("extrapolation MSE")
        ax.set_title(param_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.text(
            0.03,
            0.97,
            (
                f"r={log_log_correlation(all_x, all_y):.3f}\n"
                f"rho={spearman(all_x, all_y):.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-Parameter Relative Error vs Extrapolation", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_subset_mechanism_scatter(
    rows,
    output_path,
    x_metric,
    y_metric,
    xlabel,
    ylabel,
    title,
    annotation,
    log_x=True,
    log_y=True,
):
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    for method in sorted_methods(rows):
        group = [row for row in rows if row.get("training_method") == method]
        x = np.array([parse_float(row.get(x_metric)) for row in group], dtype=float)
        y = np.array([parse_float(row.get(y_metric)) for row in group], dtype=float)
        if log_x and log_y:
            mask = positive_mask(x, y)
        else:
            mask = np.isfinite(x) & np.isfinite(y)
            if log_x:
                mask &= x > 0
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                color=METHOD_COLORS.get(method, "gray"),
                alpha=0.78,
                s=36,
                label=METHOD_LABELS.get(method, method),
            )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    if rows:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_parameter_vs_extrapolation_matched_vf(rows, output_path):
    matched_rows = filter_rows_by_matched_vf(rows)
    if len(matched_rows) < 3:
        return
    _, _, band_label = resolve_matched_vf_band()
    vf_all = [parse_float(row.get("model_vf_rel_error")) for row in rows]
    extrap_all = [parse_float(row.get("extrapolate_mse")) for row in rows]
    overall_vf_extrap_r = log_log_correlation(vf_all, extrap_all)
    annotation = (
        "Matched vector-field band:\n"
        f"{band_label}\n"
        f"{subset_correlation_lines(matched_rows, 'parameter_rel_error', 'extrapolate_mse')}\n"
        f"overall VF→extrap: log-log r = {overall_vf_extrap_r:.3f}"
    )
    plot_subset_mechanism_scatter(
        matched_rows,
        output_path,
        "parameter_rel_error",
        "extrapolate_mse",
        r"$E_\theta=\sqrt{\sum_j\left(\frac{\hat\theta_j-\theta_j}{\theta_j}\right)^2}$",
        "extrapolation MSE",
        "Matched Vector-Field Analysis: Parameter Recovery vs Extrapolation",
        annotation,
    )


def plot_parameter_vs_extrapolation_vf_binned(rows, output_path):
    fig, axes = plt.subplots(1, len(VF_ERROR_BINS), figsize=(5.5 * len(VF_ERROR_BINS), 5.0), squeeze=False)
    for ax, (low, high, label) in zip(axes[0], VF_ERROR_BINS):
        band_rows = filter_rows_by_vf_band(rows, low, high)
        for method in sorted_methods(band_rows):
            group = [row for row in band_rows if row.get("training_method") == method]
            x = np.array([parse_float(row.get("parameter_rel_error")) for row in group], dtype=float)
            y = np.array([parse_float(row.get("extrapolate_mse")) for row in group], dtype=float)
            mask = positive_mask(x, y)
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=METHOD_COLORS.get(method, "gray"),
                    alpha=0.78,
                    s=34,
                    label=METHOD_LABELS.get(method, method),
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_\theta$")
        ax.set_ylabel("extrapolation MSE")
        ax.set_title(f"{label}\n(n={len(band_rows)})")
        ax.grid(True, which="both", alpha=0.3)
        if band_rows:
            ax.text(
                0.03,
                0.97,
                subset_correlation_lines(band_rows, "parameter_rel_error", "extrapolate_mse"),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
            )

    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Parameter Error vs Extrapolation Within Vector-Field Error Bins",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_residual_extrapolation_vs_parameter(rows, output_path):
    intercept, slope = _VF_EXTRAP_FIT
    annotation = (
        f"Fit: log10 E_extrap = {intercept:.3f} + {slope:.3f} log10 E_VF\n"
        f"{subset_correlation_lines(rows, 'parameter_rel_error', 'extrapolate_log_residual', log_y=False)}"
    )
    plot_subset_mechanism_scatter(
        rows,
        output_path,
        "parameter_rel_error",
        "extrapolate_log_residual",
        r"$E_\theta$",
        r"residual of $\log_{10} E_{\mathrm{extrap}}$ after regressing on $\log_{10} E_{\mathrm{VF}}$",
        "Parameter Recovery Does Not Explain Residual Extrapolation Error",
        annotation,
        log_x=True,
        log_y=False,
    )


def generate_main_figures(rows, main_dir, trained_models=None):
    plot_solver_ratio_metric(
        rows,
        main_dir / "fig01_extrapolation_vs_ratio_by_solver.png",
        "extrapolate_mse",
        "extrapolation MSE",
        "Extrapolation Error Across Training Solvers and Step Sizes",
    )
    plot_solver_ratio_metric(
        rows,
        main_dir / "fig02_vector_field_error_vs_ratio_by_solver.png",
        "model_vf_rel_error",
        "model vector-field relative error",
        "Vector-Field Recovery Across Training Solvers and Step Sizes",
    )
    plot_mechanism_scatter(
        rows,
        main_dir / "fig03_vector_field_vs_extrapolation.png",
        "model_vf_rel_error",
        "extrapolate_mse",
        "model vector-field relative error",
        "extrapolation MSE",
        "Vector-Field Recovery Predicts Long-Term Extrapolation",
        marker_by_reg=True,
    )
    plot_mechanism_scatter(
        rows,
        main_dir / "fig04_parameter_error_vs_vector_field_error.png",
        "parameter_rel_error",
        "model_vf_rel_error",
        r"$E_\theta=\sqrt{\sum_j\left(\frac{\hat\theta_j-\theta_j}{\theta_j}\right)^2}$",
        "model vector-field relative error",
        "Physics Parameter Recovery Does Not Necessarily Determine Vector-Field Recovery",
    )
    plot_mechanism_scatter(
        rows,
        main_dir / "fig05_parameter_error_vs_extrapolation.png",
        "parameter_rel_error",
        "extrapolate_mse",
        r"$E_\theta=\sqrt{\sum_j\left(\frac{\hat\theta_j-\theta_j}{\theta_j}\right)^2}$",
        "extrapolation MSE",
        "Physics Parameter Recovery Is a Weak Predictor of Extrapolation",
    )
    plot_parameter_vs_extrapolation_vf_binned(
        rows,
        main_dir / "fig07_parameter_vs_extrapolation_vf_binned.png",
    )
    plot_residual_extrapolation_vs_parameter(
        rows,
        main_dir / "fig08_residual_extrapolation_vs_parameter.png",
    )
    if trained_models:
        plot_representative_trajectories(
            rows,
            trained_models,
            main_dir / "fig09_representative_trajectories.png",
        )
        plot_representative_phase_portraits(
            rows,
            trained_models,
            main_dir / "fig10_representative_phase_portraits.png",
        )
    plot_parameter_recovery_panels(rows, main_dir / "fig11_physics_parameter_recovery.png")
    plot_mechanism_scatter(
        rows,
        main_dir / "fig12_best_loss_vs_extrapolation.png",
        "best_loss",
        "extrapolate_mse",
        "best validation loss",
        "extrapolation MSE",
        "Training Loss Is an Imperfect Predictor of Extrapolation",
    )
    plot_mechanism_scatter(
        rows,
        main_dir / "fig13_validation_vs_extrapolation.png",
        "validation_mse",
        "extrapolate_mse",
        "validation MSE",
        "extrapolation MSE",
        "Validation Accuracy and Long-Term Extrapolation",
    )
    plot_success_rate_heatmaps(rows, main_dir / "fig14_extrapolation_success_rate.png")
    plot_extrapolation_heatmap(rows, main_dir / "fig15_extrapolation_heatmap_solver_ratio.png")
    plot_regularization_summary(rows, main_dir / "fig16_regularization_effect.png")


def generate_appendix_figures(rows, appendix_dir, details):
    plot_within_solver_panels(
        rows,
        appendix_dir / "appendix_a01_vector_field_vs_extrapolation_by_solver.png",
        "model_vf_rel_error",
        "extrapolate_mse",
        "model vector-field relative error",
        "extrapolation MSE",
        "Vector-Field Error vs Extrapolation by Solver",
    )
    plot_within_solver_panels(
        rows,
        appendix_dir / "appendix_a02_parameter_error_vs_vector_field_by_solver.png",
        "parameter_rel_error",
        "model_vf_rel_error",
        r"$E_\theta$",
        "model vector-field relative error",
        "Parameter Error vs Vector-Field Error by Solver",
    )
    plot_within_solver_panels(
        rows,
        appendix_dir / "appendix_a03_parameter_error_vs_extrapolation_by_solver.png",
        "parameter_rel_error",
        "extrapolate_mse",
        r"$E_\theta$",
        "extrapolation MSE",
        "Parameter Error vs Extrapolation by Solver",
    )
    plot_per_parameter_extrapolation(
        rows,
        appendix_dir / "appendix_a04_per_parameter_error_vs_extrapolation.png",
    )
    plot_mechanism_scatter(
        rows,
        appendix_dir / "appendix_a05_residual_error_vs_extrapolation.png",
        "residual_rel_error",
        "extrapolate_mse",
        "residual relative error",
        "extrapolation MSE",
        "Residual Recovery vs Extrapolation",
    )
    plot_regularization_solver_interaction(
        rows,
        appendix_dir / "appendix_a06_regularization_solver_interaction.png",
    )
    plot_seed_stability(rows, appendix_dir / "appendix_a07_seed_stability.png")
    plot_error_over_time(details, rows, appendix_dir / "appendix_a08_error_vs_time.png")
    plot_fixed_ratio_correlation_heatmaps(
        rows,
        appendix_dir / "appendix_a09_fixed_ratio_correlations.png",
    )
    plot_parameter_vs_extrapolation_matched_vf(
        rows,
        appendix_dir / "appendix_a10_matched_vf_parameter_vs_extrapolation.png",
    )


def visualize(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    main_dir = output_dir / "main"
    appendix_dir = output_dir / "appendix"
    tables_dir = output_dir / "tables"
    for path in (main_dir, appendix_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(input_dir / "carrying_big_factor_sweep_summary.csv")
    rows = add_derived_metrics(rows)
    payload = load_pickle(input_dir / "carrying_big_factor_sweep_details.pkl")
    details = payload.get("details", {})
    trained_models = payload.get("trained", {})

    write_group_summary(rows, tables_dir)
    write_correlation_summary(rows, tables_dir)
    write_conditional_analysis_summary(rows, tables_dir)

    generate_main_figures(rows, main_dir, trained_models)
    generate_appendix_figures(rows, appendix_dir, details)

    print(f"Saved main figures to {main_dir}")
    print(f"Saved appendix figures to {appendix_dir}")
    print(f"Saved tables to {tables_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for the carrying big factor sweep."
    )
    parser.add_argument("--input-dir", default="carrying_big_factor_sweep_results")
    parser.add_argument("--output-dir", default="carrying_big_factor_sweep_visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
