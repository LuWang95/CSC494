import argparse
import csv
import pickle
from pathlib import Path

from carrying_solver_sweep import (
    evaluate,
    forward_euler,
    heun,
    make_reference_data,
    progress,
    rk4,
    train_one,
    write_summary_csv,
)
from experiment_metadata import write_run_manifest


METHODS = {
    "forward_euler": forward_euler,
    "heun": heun,
    "rk4": rk4,
}

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


def build_training_configs(method_names, train_ratios, include_diffrax):
    configs = []
    for method_name in method_names:
        method = METHODS[method_name]
        for ratio in train_ratios:
            configs.append((f"{method_name}_ratio{ratio}", method, ratio))
    if include_diffrax:
        configs.append(("diffrax_tsit5", None, "adaptive"))
    return configs


def load_existing_results(output_dir):
    summary_path = output_dir / "carrying_big_factor_sweep_summary.csv"
    details_path = output_dir / "carrying_big_factor_sweep_details.pkl"
    rows = []
    details = {}
    trained_models = {}

    if summary_path.exists():
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    if details_path.exists():
        with details_path.open("rb") as f:
            payload = pickle.load(f)
        details = payload.get("details", {})
        trained_models = payload.get("trained", {})

    completed = {
        (
            row.get("training_config"),
            row.get("regularization_profile"),
            int(row.get("seed")),
        )
        for row in rows
        if row.get("training_config")
        and row.get("regularization_profile")
        and row.get("seed") not in (None, "")
    }
    return rows, details, trained_models, completed


def save_results(output_dir, rows, details, trained_models, training_configs, reg_profiles):
    write_summary_csv(rows, output_dir / "carrying_big_factor_sweep_summary.csv")
    with (output_dir / "carrying_big_factor_sweep_details.pkl").open("wb") as f:
        pickle.dump(
            {
                "trained": trained_models,
                "rows": rows,
                "details": details,
                "training_configs": [
                    (name, method.__name__ if method is not None else "diffrax_tsit5", ratio)
                    for name, method, ratio in training_configs
                ],
                "regularization_profiles": reg_profiles,
            },
            f,
        )


def run_big_sweep(
    output_dir,
    seeds,
    method_names,
    train_ratios,
    reg_profile_names,
    include_diffrax,
    chunk_size,
    stage_scale,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_configs = build_training_configs(
        method_names,
        train_ratios,
        include_diffrax,
    )
    reg_profiles = [REGULARIZATION_PROFILES[name] for name in reg_profile_names]
    write_run_manifest(
        output_dir,
        "carrying_big_factor_sweep",
        {
            "seeds": list(seeds),
            "methods": list(method_names),
            "train_ratios": list(train_ratios),
            "regularization_profiles": list(reg_profile_names),
            "include_diffrax": include_diffrax,
            "chunk_size": chunk_size,
            "stage_scale": stage_scale,
        },
    )

    progress(f"[setup] writing big factor sweep to {output_dir}")
    progress(
        f"[setup] configs={len(training_configs)}, "
        f"regularization_profiles={len(reg_profiles)}, seeds={len(seeds)}"
    )
    progress("[setup] building reference data")
    data = make_reference_data()
    progress("[setup] reference data ready")

    rows, details, trained_models, completed = load_existing_results(output_dir)
    if completed:
        progress(f"[resume] found {len(completed)} completed runs; skipping them")

    total = len(training_configs) * len(reg_profiles) * len(seeds)
    run_idx = 0
    for config_name, method, ratio in training_configs:
        for reg_profile in reg_profiles:
            for seed in seeds:
                run_idx += 1
                key = (config_name, reg_profile["name"], seed)
                if key in completed:
                    progress(
                        f"[run {run_idx}/{total}] skip completed "
                        f"{config_name}, reg={reg_profile['name']}, seed={seed}"
                    )
                    continue

                progress(
                    f"[run {run_idx}/{total}] train {config_name}, "
                    f"reg={reg_profile['name']}, seed={seed}"
                )
                trained = train_one(
                    seed=seed,
                    config_name=config_name,
                    method=method,
                    ratio=ratio,
                    data=data,
                    chunk_size=chunk_size,
                    stage_scale=stage_scale,
                    regularization_profile=reg_profile,
                )
                trained_models[key] = trained

                progress(
                    f"[run {run_idx}/{total}] eval {config_name}, "
                    f"reg={reg_profile['name']}, seed={seed}, "
                    f"best_loss={trained['best_loss']:.6e}"
                )
                row, curves = evaluate(trained, data)
                rows.append(row)
                details[key] = curves
                progress(
                    f"[run {run_idx}/{total}] done {config_name}, "
                    f"reg={reg_profile['name']}, seed={seed}, "
                    f"val={row['validation_mse']:.6e}, "
                    f"extrap={row['extrapolate_mse']:.6e}, "
                    f"residual={row['residual_mse']:.6e}"
                )
                save_results(output_dir, rows, details, trained_models, training_configs, reg_profiles)

    save_results(output_dir, rows, details, trained_models, training_configs, reg_profiles)
    progress("[done] carrying big factor sweep complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Large carrying-capacity sweep over solver, ratio, seed, and representative "
            "regularization settings."
        )
    )
    parser.add_argument("--output-dir", default="carrying_big_factor_sweep_results")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["forward_euler", "heun", "rk4"],
    )
    parser.add_argument("--train-ratios", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--regularization-profiles",
        nargs="+",
        choices=sorted(REGULARIZATION_PROFILES),
        default=["none", "l2_small", "ortho_small", "l2_plus_ortho"],
    )
    parser.add_argument(
        "--no-diffrax",
        action="store_true",
        help="Disable the adaptive Diffrax training configuration.",
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--stage-scale",
        type=float,
        default=1.0,
        help="Scale stage epochs/patience for quick smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_big_sweep(
        output_dir=args.output_dir,
        seeds=args.seeds,
        method_names=args.methods,
        train_ratios=args.train_ratios,
        reg_profile_names=args.regularization_profiles,
        include_diffrax=not args.no_diffrax,
        chunk_size=args.chunk_size,
        stage_scale=args.stage_scale,
    )
