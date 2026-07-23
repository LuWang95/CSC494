import argparse
import csv
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from carrying_solver_sweep import (
    progress,
    solve_model_diffrax_batch,
    vector_field_metrics,
    y0_batch,
)
from carrying_train_interval_sweep import (
    COVERAGE_EVAL_HORIZON,
    DEFAULT_COVERAGE_RADIUS,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_RATIO,
    DEFAULT_TRAIN_INTERVALS,
    EVALUATION_SCHEMA_VERSION,
    FIXED_EXTRAP_HORIZONS,
    OBS_DT,
    SUMMARY_FIELDS as INTERVAL_SUMMARY_FIELDS,
    T_FINAL,
    TRAINING_PROTOCOL_VERSION,
    candidate_horizon_mse,
    evaluate_interval,
    evaluation_failure_row,
    horizon_field_name,
    horizon_mse,
    make_reference_data_for_interval,
    train_one_interval,
    validate_sweep_args,
)
from experiment_metadata import write_run_manifest


EXPERIMENT_KIND = "carrying_parameter_importance_sweep"
PARAMETER_IMPORTANCE_PROTOCOL_VERSION = 1
TRUE_PHYSICS = jnp.array([1.0, 0.05, 1.5, 0.03])

REGULARIZATION_PROFILES = {
    "none": {
        "name": "none",
        "l2_weight": 0.0,
        "ortho_weight": 0.0,
    },
    "l2_small": {
        "name": "l2_small",
        "l2_weight": 1e-5,
        "ortho_weight": 0.0,
    },
    "ortho_small": {
        "name": "ortho_small",
        "l2_weight": 0.0,
        "ortho_weight": 1e-3,
    },
    "l2_plus_ortho": {
        "name": "l2_plus_ortho",
        "l2_weight": 1e-5,
        "ortho_weight": 1e-3,
    },
}

IMPORTANCE_FIELDS = [
    "parameter_importance_protocol_version",
    "parameter_rel_error",
    "parameter_componentwise_rel_rmse",
    "oracle_evaluation_status",
    "oracle_evaluation_error_type",
    "oracle_evaluation_error",
    "oracle_model_vf_rel_error",
]
for horizon in FIXED_EXTRAP_HORIZONS:
    label = f"{horizon:g}".replace(".", "p")
    IMPORTANCE_FIELDS.extend(
        [
            f"oracle_same_ic_extrap_mse_h{label}",
            f"oracle_candidate_ic_extrap_mse_h{label}",
            f"oracle_gain_same_ic_h{label}",
            f"oracle_gain_candidate_ic_h{label}",
        ]
    )

SUMMARY_FIELDS = list(dict.fromkeys(INTERVAL_SUMMARY_FIELDS + IMPORTANCE_FIELDS))


def parse_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def parse_int(value, default=None):
    number = parse_float(value)
    return int(number) if np.isfinite(number) else default


def run_key(
    train_interval,
    ratio,
    noise_level,
    coverage_radius,
    seed,
    chunk_size,
    stage_scale,
    regularization_profile,
):
    return (
        float(train_interval),
        int(ratio),
        float(noise_level),
        float(coverage_radius),
        int(seed),
        int(chunk_size),
        float(stage_scale),
        int(TRAINING_PROTOCOL_VERSION),
        str(regularization_profile),
        int(PARAMETER_IMPORTANCE_PROTOCOL_VERSION),
    )


def row_run_key(row):
    return run_key(
        parse_float(row.get("train_interval")),
        parse_int(row.get("ratio"), DEFAULT_RATIO),
        parse_float(row.get("noise_level"), DEFAULT_NOISE_LEVEL),
        parse_float(row.get("coverage_radius"), DEFAULT_COVERAGE_RADIUS),
        parse_int(row.get("seed")),
        parse_int(row.get("chunk_size")),
        parse_float(row.get("stage_scale")),
        row.get("regularization_profile", "none"),
    )


def parameter_metrics(params):
    learned = params["f_physics"]
    global_relative = jnp.linalg.norm(learned - TRUE_PHYSICS) / jnp.linalg.norm(
        TRUE_PHYSICS
    )
    componentwise = jnp.sqrt(
        jnp.mean(((learned - TRUE_PHYSICS) / (jnp.abs(TRUE_PHYSICS) + 1e-12)) ** 2)
    )
    return float(global_relative), float(componentwise)


def safe_gain(base_value, oracle_value):
    if not np.isfinite(base_value) or not np.isfinite(oracle_value):
        return np.nan
    return float(base_value / (oracle_value + 1e-12))


def oracle_intervention_metrics(trained, data):
    oracle_params = {**trained["params"], "f_physics": TRUE_PHYSICS}
    t_final_grid = data["t_final_grid"]
    extrap_start = data["extrap_start"]
    pred_same = solve_model_diffrax_batch(
        y0_batch,
        oracle_params,
        t_final_grid,
        data["T_final"],
    )
    pred_candidates = solve_model_diffrax_batch(
        data["y0_candidates"],
        oracle_params,
        t_final_grid,
        data["T_final"],
    )

    metrics = {
        "oracle_evaluation_status": "ok",
        "oracle_evaluation_error_type": "",
        "oracle_evaluation_error": "",
        "oracle_model_vf_rel_error": float(
            vector_field_metrics(oracle_params)["model_vf_rel_error"]
        ),
    }
    for horizon in FIXED_EXTRAP_HORIZONS:
        label = f"{horizon:g}".replace(".", "p")
        metrics[f"oracle_same_ic_extrap_mse_h{label}"] = horizon_mse(
            pred_same,
            data["extrap"],
            extrap_start,
            horizon,
            data["obs_dt"],
        )
        candidate_values = candidate_horizon_mse(
            pred_candidates,
            data["candidate_clean"],
            extrap_start,
            horizon,
            data["obs_dt"],
        )
        metrics[f"oracle_candidate_ic_extrap_mse_h{label}"] = float(
            jnp.mean(candidate_values)
        )
    return metrics


def add_importance_metrics(row, trained, data):
    parameter_rel, componentwise_rel = parameter_metrics(trained["params"])
    row["parameter_importance_protocol_version"] = (
        PARAMETER_IMPORTANCE_PROTOCOL_VERSION
    )
    row["parameter_rel_error"] = parameter_rel
    row["parameter_componentwise_rel_rmse"] = componentwise_rel

    try:
        oracle = oracle_intervention_metrics(trained, data)
    except Exception as error:
        message = " ".join(str(error).split())
        oracle = {
            "oracle_evaluation_status": "failed",
            "oracle_evaluation_error_type": type(error).__name__,
            "oracle_evaluation_error": message[:1000],
            "oracle_model_vf_rel_error": np.nan,
        }
    row.update(oracle)

    for horizon in FIXED_EXTRAP_HORIZONS:
        label = f"{horizon:g}".replace(".", "p")
        same_field = horizon_field_name(horizon)
        candidate_field = f"candidate_ic_extrap_mse_h{label}"
        oracle_same_field = f"oracle_same_ic_extrap_mse_h{label}"
        oracle_candidate_field = f"oracle_candidate_ic_extrap_mse_h{label}"
        row[f"oracle_gain_same_ic_h{label}"] = safe_gain(
            parse_float(row.get(same_field)),
            parse_float(row.get(oracle_same_field)),
        )
        row[f"oracle_gain_candidate_ic_h{label}"] = safe_gain(
            parse_float(row.get(candidate_field)),
            parse_float(row.get(oracle_candidate_field)),
        )
    return row


def load_existing_results(output_dir):
    summary_path = output_dir / "carrying_parameter_importance_sweep_summary.csv"
    details_path = output_dir / "carrying_parameter_importance_sweep_details.pkl"
    rows, details, trained_models = [], {}, {}
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    if details_path.exists():
        with details_path.open("rb") as f:
            payload = pickle.load(f)
        details = payload.get("details", {})
        trained_models = payload.get("trained", {})

    def compatible_key(key):
        return (
            isinstance(key, tuple)
            and len(key) == 10
            and int(key[7]) == TRAINING_PROTOCOL_VERSION
            and int(key[9]) == PARAMETER_IMPORTANCE_PROTOCOL_VERSION
        )

    rows = [
        row
        for row in rows
        if row.get("experiment_kind") == EXPERIMENT_KIND
        and parse_int(row.get("evaluation_schema_version"))
        == EVALUATION_SCHEMA_VERSION
        and parse_int(row.get("training_protocol_version"))
        == TRAINING_PROTOCOL_VERSION
        and parse_int(row.get("parameter_importance_protocol_version"))
        == PARAMETER_IMPORTANCE_PROTOCOL_VERSION
    ]
    details = {key: value for key, value in details.items() if compatible_key(key)}
    trained_models = {
        key: value for key, value in trained_models.items() if compatible_key(key)
    }
    completed = {
        row_run_key(row)
        for row in rows
        if row.get("evaluation_status") == "ok"
    }
    return rows, details, trained_models, completed


def save_results(
    output_dir,
    rows,
    details,
    trained_models,
    manifest_parameters,
):
    with (output_dir / "carrying_parameter_importance_sweep_summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "carrying_parameter_importance_sweep_details.pkl").open(
        "wb"
    ) as f:
        pickle.dump(
            {
                "trained": trained_models,
                "rows": rows,
                "details": details,
                "manifest_parameters": manifest_parameters,
            },
            f,
        )


def run_parameter_importance_sweep(
    output_dir,
    seeds,
    train_intervals,
    ratio,
    noise_level,
    coverage_radius,
    regularization_profile_names,
    chunk_size,
    stage_scale,
):
    validate_sweep_args(
        train_intervals,
        ratio,
        noise_level,
        coverage_radius,
        chunk_size,
        stage_scale,
    )
    unknown_profiles = set(regularization_profile_names) - set(
        REGULARIZATION_PROFILES
    )
    if unknown_profiles:
        raise ValueError(f"Unknown regularization profiles: {sorted(unknown_profiles)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_parameters = {
        "seeds": list(seeds),
        "train_intervals": list(train_intervals),
        "ratio": ratio,
        "noise_level": noise_level,
        "coverage_radius": coverage_radius,
        "coverage_eval_horizon": COVERAGE_EVAL_HORIZON,
        "fixed_extrap_horizons": FIXED_EXTRAP_HORIZONS,
        "regularization_profiles": list(regularization_profile_names),
        "chunk_size": chunk_size,
        "stage_scale": stage_scale,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "parameter_importance_protocol_version": (
            PARAMETER_IMPORTANCE_PROTOCOL_VERSION
        ),
        "obs_dt": OBS_DT,
        "t_final": T_FINAL,
    }
    write_run_manifest(
        output_dir,
        EXPERIMENT_KIND,
        manifest_parameters,
        filename="run_manifest.json",
    )
    write_run_manifest(
        output_dir,
        f"{EXPERIMENT_KIND}_evaluation",
        manifest_parameters,
        filename="evaluation_manifest.json",
    )

    rows, details, trained_models, completed = load_existing_results(output_dir)
    training_manifest_path = output_dir / "training_manifest.json"
    if not training_manifest_path.exists() and not trained_models:
        write_run_manifest(
            output_dir,
            f"{EXPERIMENT_KIND}_training",
            manifest_parameters,
            filename="training_manifest.json",
        )
    elif not training_manifest_path.exists():
        progress(
            "[warning] compatible models exist without training provenance; "
            "no training manifest was fabricated"
        )

    total = len(train_intervals) * len(regularization_profile_names) * len(seeds)
    run_index = 0
    for train_interval in train_intervals:
        progress(f"[setup] reference data for T_train={train_interval:g}")
        data = make_reference_data_for_interval(
            train_interval,
            noise_level=noise_level,
            coverage_radius=coverage_radius,
        )
        data["noise_level"] = noise_level
        for profile_name in regularization_profile_names:
            profile = REGULARIZATION_PROFILES[profile_name]
            for seed in seeds:
                run_index += 1
                key = run_key(
                    train_interval,
                    ratio,
                    noise_level,
                    coverage_radius,
                    seed,
                    chunk_size,
                    stage_scale,
                    profile_name,
                )
                if key in completed:
                    progress(
                        f"[run {run_index}/{total}] skip T={train_interval:g}, "
                        f"reg={profile_name}, seed={seed}"
                    )
                    continue

                if key in trained_models:
                    progress(
                        f"[run {run_index}/{total}] reuse T={train_interval:g}, "
                        f"reg={profile_name}, seed={seed}"
                    )
                    trained = trained_models[key]
                else:
                    progress(
                        f"[run {run_index}/{total}] train T={train_interval:g}, "
                        f"reg={profile_name}, seed={seed}"
                    )
                    trained = train_one_interval(
                        seed=seed,
                        train_interval=train_interval,
                        ratio=ratio,
                        data=data,
                        chunk_size=chunk_size,
                        stage_scale=stage_scale,
                        regularization_profile=profile,
                    )
                    trained_models[key] = trained
                    save_results(
                        output_dir,
                        rows,
                        details,
                        trained_models,
                        manifest_parameters,
                    )

                try:
                    row, curves = evaluate_interval(trained, data)
                    row["experiment_kind"] = EXPERIMENT_KIND
                    row = add_importance_metrics(row, trained, data)
                except Exception as error:
                    row = evaluation_failure_row(trained, data, error)
                    row["experiment_kind"] = EXPERIMENT_KIND
                    row["parameter_importance_protocol_version"] = (
                        PARAMETER_IMPORTANCE_PROTOCOL_VERSION
                    )
                    parameter_rel, componentwise_rel = parameter_metrics(
                        trained["params"]
                    )
                    row["parameter_rel_error"] = parameter_rel
                    row["parameter_componentwise_rel_rmse"] = componentwise_rel
                    curves = None
                    progress(
                        f"[run {run_index}/{total}] evaluation failed: "
                        f"{row['evaluation_error_type']}: {row['evaluation_error']}"
                    )

                rows = [existing for existing in rows if row_run_key(existing) != key]
                rows.append(row)
                if curves is None:
                    details.pop(key, None)
                else:
                    details[key] = curves
                save_results(
                    output_dir,
                    rows,
                    details,
                    trained_models,
                    manifest_parameters,
                )

    progress(f"[done] parameter-importance sweep complete: {total} requested runs")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether physics-parameter recovery matters more at shorter "
            "training intervals, using regularization variation and an oracle "
            "parameter-replacement intervention."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="carrying_parameter_importance_sweep_results",
    )
    parser.add_argument(
        "--train-intervals",
        nargs="+",
        type=float,
        default=DEFAULT_TRAIN_INTERVALS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--ratio", type=int, default=DEFAULT_RATIO)
    parser.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL)
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS,
    )
    parser.add_argument(
        "--regularization-profiles",
        nargs="+",
        choices=sorted(REGULARIZATION_PROFILES),
        default=list(REGULARIZATION_PROFILES),
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--stage-scale", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_parameter_importance_sweep(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_intervals=args.train_intervals,
        ratio=args.ratio,
        noise_level=args.noise_level,
        coverage_radius=args.coverage_radius,
        regularization_profile_names=args.regularization_profiles,
        chunk_size=args.chunk_size,
        stage_scale=args.stage_scale,
    )
