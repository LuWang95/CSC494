import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


OUTCOMES = [
    "same_ic_extrap_mse_h2p5",
    "same_ic_extrap_mse_h5",
    "same_ic_extrap_mse_h10",
    "candidate_ic_extrap_mse_h2p5",
    "candidate_ic_extrap_mse_h5",
    "candidate_ic_extrap_mse_h10",
]


def parse_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def read_successful_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("evaluation_status") == "ok"]


def standardize(values):
    values = np.asarray(values, dtype=float)
    scale = np.std(values)
    if not np.isfinite(scale) or scale == 0.0:
        return np.full_like(values, np.nan)
    return (values - np.mean(values)) / scale


def residualize(values, controls):
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    design = np.column_stack([np.ones(values.size), controls])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def regularization_dummies(profiles):
    profiles = np.asarray(profiles)
    names = sorted(set(profiles))
    baseline = "none" if "none" in names else names[0]
    dummy_names = [name for name in names if name != baseline]
    if not dummy_names:
        return np.empty((profiles.size, 0))
    return np.column_stack([profiles == name for name in dummy_names]).astype(float)


def partial_spearman(x, y, vf_control, profile_control):
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    controls = np.column_stack(
        [rankdata(vf_control), regularization_dummies(profile_control)]
    )
    x_residual = residualize(x_rank, controls)
    y_residual = residualize(y_rank, controls)
    if np.std(x_residual) == 0.0 or np.std(y_residual) == 0.0:
        return np.nan
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def interval_regression(parameter_error, vf_error, outcome, profiles):
    y = standardize(np.log10(outcome))
    parameter = standardize(np.log10(parameter_error))
    vf = standardize(np.log10(vf_error))
    if not np.all(np.isfinite(np.concatenate([y, parameter, vf]))):
        return np.nan, np.nan, np.nan
    design = np.column_stack(
        [np.ones(y.size), parameter, vf, regularization_dummies(profiles)]
    )
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coefficients
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - np.sum((y - prediction) ** 2) / total if total > 0 else np.nan
    return float(coefficients[1]), float(coefficients[2]), float(r_squared)


def outcome_arrays(rows, outcome):
    parameter = np.asarray([parse_float(row.get("parameter_rel_error")) for row in rows])
    vf = np.asarray([parse_float(row.get("model_vf_rel_error")) for row in rows])
    y = np.asarray([parse_float(row.get(outcome)) for row in rows])
    mask = (
        np.isfinite(parameter)
        & np.isfinite(vf)
        & np.isfinite(y)
        & (parameter > 0)
        & (vf > 0)
        & (y > 0)
    )
    return parameter[mask], vf[mask], y[mask], mask


def oracle_gain_field(outcome):
    if outcome.startswith("same_ic"):
        return outcome.replace("same_ic_extrap_mse_", "oracle_gain_same_ic_")
    return outcome.replace(
        "candidate_ic_extrap_mse_",
        "oracle_gain_candidate_ic_",
    )


def summarize_by_interval(rows):
    output = []
    train_intervals = sorted({parse_float(row.get("train_interval")) for row in rows})
    for train_interval in train_intervals:
        group = [
            row
            for row in rows
            if parse_float(row.get("train_interval")) == train_interval
        ]
        for outcome in OUTCOMES:
            parameter, vf, y, mask = outcome_arrays(group, outcome)
            if y.size < 4:
                continue
            selected_profiles = np.asarray(
                [
                    row.get("regularization_profile", "none")
                    for row, keep in zip(group, mask)
                    if keep
                ]
            )
            beta_parameter, beta_vf, r_squared = interval_regression(
                parameter,
                vf,
                y,
                selected_profiles,
            )
            gain_field = oracle_gain_field(outcome)
            gain = np.asarray(
                [parse_float(row.get(gain_field)) for row, keep in zip(group, mask) if keep]
            )
            gain = gain[np.isfinite(gain) & (gain > 0)]
            output.append(
                {
                    "train_interval": train_interval,
                    "outcome": outcome,
                    "n": y.size,
                    "parameter_extrap_spearman": float(
                        spearmanr(parameter, y).correlation
                    ),
                    "vf_extrap_spearman": float(spearmanr(vf, y).correlation),
                    "parameter_extrap_partial_spearman_given_vf_and_reg": partial_spearman(
                        parameter,
                        y,
                        vf,
                        selected_profiles,
                    ),
                    "standardized_beta_parameter_given_vf": beta_parameter,
                    "standardized_beta_vf_given_parameter": beta_vf,
                    "regression_r_squared": r_squared,
                    "oracle_gain_median": (
                        float(np.median(gain)) if gain.size else np.nan
                    ),
                    "oracle_gain_q25": (
                        float(np.percentile(gain, 25)) if gain.size else np.nan
                    ),
                    "oracle_gain_q75": (
                        float(np.percentile(gain, 75)) if gain.size else np.nan
                    ),
                }
            )
    return output


def interaction_coefficient(rows, outcome):
    parameter, vf, y, mask = outcome_arrays(rows, outcome)
    selected = [row for row, keep in zip(rows, mask) if keep]
    if y.size < 8:
        return np.nan, np.nan, np.nan, y.size

    train_interval = standardize(
        [parse_float(row.get("train_interval")) for row in selected]
    )
    parameter = standardize(np.log10(parameter))
    vf = standardize(np.log10(vf))
    y = standardize(np.log10(y))
    regularizations = sorted(
        {row.get("regularization_profile", "none") for row in selected}
    )
    baseline = "none" if "none" in regularizations else regularizations[0]
    dummy_names = [name for name in regularizations if name != baseline]
    dummies = np.column_stack(
        [
            [float(row.get("regularization_profile", "none") == name) for row in selected]
            for name in dummy_names
        ]
    ) if dummy_names else np.empty((y.size, 0))
    design = np.column_stack(
        [
            np.ones(y.size),
            train_interval,
            parameter,
            vf,
            parameter * train_interval,
            dummies,
        ]
    )
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coefficients
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - np.sum((y - prediction) ** 2) / total if total > 0 else np.nan
    return float(coefficients[4]), float(coefficients[2]), float(r_squared), y.size


def bootstrap_interaction(rows, outcome, samples, seed):
    rng = np.random.default_rng(seed)
    seed_values = sorted({int(row["seed"]) for row in rows})
    estimates = []
    for _ in range(samples):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        sample_rows = []
        for sampled_seed in sampled_seeds:
            sample_rows.extend(
                row for row in rows if int(row["seed"]) == int(sampled_seed)
            )
        estimate, _, _, _ = interaction_coefficient(sample_rows, outcome)
        if np.isfinite(estimate):
            estimates.append(estimate)
    estimates = np.asarray(estimates)
    if estimates.size == 0:
        return np.nan, np.nan, np.nan
    low, high = np.percentile(estimates, [2.5, 97.5])
    p_two_sided = 2.0 * min(np.mean(estimates <= 0), np.mean(estimates >= 0))
    return float(low), float(high), float(min(1.0, p_two_sided))


def summarize_interactions(rows, bootstrap_samples, bootstrap_seed):
    output = []
    for outcome in OUTCOMES:
        interaction, main_parameter, r_squared, n = interaction_coefficient(rows, outcome)
        low, high, p_value = bootstrap_interaction(
            rows,
            outcome,
            bootstrap_samples,
            bootstrap_seed,
        )
        output.append(
            {
                "outcome": outcome,
                "n": n,
                "standardized_parameter_main_effect": main_parameter,
                "parameter_error_x_train_interval_interaction": interaction,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_two_sided_p": p_value,
                "regression_r_squared": r_squared,
                "short_interval_importance_supported": bool(
                    np.isfinite(interaction) and interaction < 0 and high < 0
                ),
            }
        )
    return output


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_successful_rows(input_path)
    if not rows:
        raise RuntimeError(f"No successful rows found in {input_path}")

    interval_rows = summarize_by_interval(rows)
    interaction_rows = summarize_interactions(
        rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    write_rows(output_dir / "parameter_importance_by_interval.csv", interval_rows)
    write_rows(output_dir / "parameter_importance_interaction_test.csv", interaction_rows)
    print(f"Saved parameter-importance analysis to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Quantify whether parameter recovery predicts extrapolation more "
            "strongly at shorter training intervals, controlling VF error."
        )
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
        default="carrying_parameter_importance_analysis",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
