import pickle
from pathlib import Path

from solver_error_no_noise import (
    evaluate_model_diffrax,
    forward_euler,
    make_reference_data,
    plot_error_vs_time,
    progress,
    train_model,
    write_summary_csv,
)


def run_euler_sweep(output_dir="no_noise_euler_diffrax_prediction_sweep_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(f"[setup] writing Euler-only results to {output_dir}")
    progress("[setup] building clean reference data with Diffrax")
    data = make_reference_data()
    progress("[setup] reference data ready")

    seeds = [0, 1, 2, 3, 4]
    train_ratios = [1, 2, 4, 8, 16]
    total_runs = len(seeds) * len(train_ratios)

    all_trained = {}
    rows = []
    details = {}

    run_idx = 0
    for train_ratio in train_ratios:
        for seed in seeds:
            run_idx += 1
            progress(
                f"[train {run_idx}/{total_runs}] "
                f"start seed={seed}, method=forward_euler, train_ratio={train_ratio}"
            )
            trained = train_model(
                seed,
                forward_euler,
                train_ratio,
                data,
                num_epochs=30000,
                step_size=3e-3,
                chunk_size=100,
            )
            all_trained[(seed, "forward_euler", train_ratio)] = trained
            progress(
                f"[train {run_idx}/{total_runs}] "
                f"done seed={seed}, method=forward_euler, "
                f"train_ratio={train_ratio}, best_loss={trained['best_loss']:.6e}"
            )

    progress("[eval] starting Diffrax/Tsit5 evaluation")
    for eval_idx, ((seed, train_method_name, train_ratio), trained) in enumerate(
        all_trained.items(),
        start=1,
    ):
        progress(
            f"[eval {eval_idx}/{total_runs}] "
            f"start seed={seed}, method={train_method_name}, "
            f"train_ratio={train_ratio}"
        )
        row, curves = evaluate_model_diffrax(
            trained=trained,
            data=data,
            experiment_type="euler_only_diffrax_eval",
        )
        rows.append(row)
        key = (
            seed,
            train_method_name,
            train_ratio,
            "diffrax_tsit5",
            "adaptive",
        )
        details[key] = curves
        progress(
            f"[eval {eval_idx}/{total_runs}] "
            f"done seed={seed}, method={train_method_name}, "
            f"train_ratio={train_ratio}, "
            f"train_mse={row['train_mse']:.6e}, "
            f"val_mse={row['validation_mse']:.6e}, "
            f"extrap_mse={row['extrapolate_mse']:.6e}, "
            f"instability={row['instability_rate']:.3f}"
        )

    progress("[save] writing summary CSV and details pickle")
    write_summary_csv(rows, output_dir / "prediction_sweep_summary.csv")
    with (output_dir / "prediction_sweep_details.pkl").open("wb") as f:
        pickle.dump(
            {
                "trained": all_trained,
                "rows": rows,
                "details": details,
            },
            f,
        )

    progress("[plot] writing error_vs_time_by_solver.png")
    plot_error_vs_time(details, output_dir)
    progress("[done] Euler-only no-noise Diffrax sweep complete")
    return {
        "trained": all_trained,
        "rows": rows,
        "details": details,
    }


if __name__ == "__main__":
    run_euler_sweep()
