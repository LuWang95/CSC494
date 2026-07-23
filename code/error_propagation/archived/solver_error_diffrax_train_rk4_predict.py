import argparse
import pickle
from pathlib import Path

import diffrax
import jax.numpy as jnp
import optax
from jax import jit, lax, value_and_grad, vmap
from jax import random

from solver_error_no_noise import (
    T,
    T_extrapolate,
    init_network_params,
    instability_rate,
    make_optimizer,
    make_reference_data,
    model_rhs,
    nn,
    noise_level,
    num_extrapolate_observed,
    num_observed,
    obs_dt,
    progress,
    rk4,
    roll_out,
    t_extrapolate,
    t_obs,
    time_error_curves,
    vector_field_metrics,
    write_summary_csv,
    y0_batch,
    y0_validation,
)


def initial_params(seed):
    return {
        "nn_params": init_network_params([2, 64, 64, 2], random.key(seed)),
        "f_physics": jnp.array([1.0, 0.05, 1.5, 0.03]),
    }


def solve_model_diffrax(y0, params, ts, t1, rtol, atol):
    def rhs(t, y, args):
        return model_rhs(y, args)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=1e-3,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=500000,
    )
    return sol.ys


def train_model_diffrax(seed, data, num_epochs, step_size, chunk_size, rtol, atol):
    params = initial_params(seed)
    optimizer = make_optimizer(step_size)
    opt_state = optimizer.init(params)

    def sample_loss(parameters, y0, target):
        pred = solve_model_diffrax(y0, parameters, t_obs, T, rtol, atol)
        return jnp.mean((target - pred) ** 2)

    def batch_loss(parameters):
        losses = vmap(sample_loss, in_axes=(None, 0, 0))(
            parameters, y0_batch, data["train_noisy"]
        )
        residuals = vmap(nn, in_axes=(0, None))(
            data["train_noisy"].reshape(-1, 2), parameters["nn_params"]
        )
        return jnp.mean(losses) + 1e-5 * jnp.mean(residuals**2)

    @jit
    def train_step(carry, _):
        parameters, opt_state = carry
        loss_val, grads = value_and_grad(batch_loss)(parameters)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        return (optax.apply_updates(parameters, updates), opt_state), loss_val

    @jit
    def train_chunk(parameters, opt_state):
        (parameters, opt_state), losses = lax.scan(
            train_step,
            (parameters, opt_state),
            None,
            length=chunk_size,
        )
        return parameters, opt_state, losses

    best_loss = float("inf")
    best_epoch = 0
    best_params = params
    patience = 500
    counter = 0

    for chunk_idx in range(num_epochs // chunk_size):
        params, opt_state, losses = train_chunk(params, opt_state)
        loss_val = float(losses[-1])
        current_epoch = (chunk_idx + 1) * chunk_size
        if best_loss - loss_val > 1e-7:
            best_loss = loss_val
            best_epoch = current_epoch
            best_params = params
            counter = 0
        else:
            counter += chunk_size
        if counter >= patience:
            break

    return {
        "params": best_params,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "training_method": "diffrax_tsit5",
        "ratio": "adaptive",
        "h_model": "adaptive",
        "seed": seed,
        "train_rtol": rtol,
        "train_atol": atol,
    }


def rollout_batch(y0s, h, rhs, params, num_steps, method):
    return vmap(lambda y0: roll_out(y0, h, rhs, params, num_steps, method))(y0s)


def evaluate_with_high_precision_rk4(trained, data, predict_ratio):
    h = obs_dt / predict_ratio
    steps = int((num_observed - 1) * predict_ratio)
    extrap_steps = int((num_extrapolate_observed - 1) * predict_ratio)
    params = trained["params"]

    pred_train = rollout_batch(y0_batch, h, model_rhs, params, steps, rk4)[
        :, ::predict_ratio, :
    ]
    pred_val = rollout_batch(y0_validation, h, model_rhs, params, steps, rk4)[
        :, ::predict_ratio, :
    ]
    pred_extrap = rollout_batch(y0_batch, h, model_rhs, params, extrap_steps, rk4)[
        :, ::predict_ratio, :
    ]

    train_mse_by_time, train_rel_by_time, train_mse_by_time_state = time_error_curves(
        pred_train, data["train_clean"]
    )
    val_mse_by_time, val_rel_by_time, val_mse_by_time_state = time_error_curves(
        pred_val, data["val_clean"]
    )
    extrap_mse_by_time, extrap_rel_by_time, extrap_mse_by_time_state = time_error_curves(
        pred_extrap, data["extrap"]
    )
    extrap_start = num_observed

    row = {
        "experiment_kind": "diffrax_train_rk4_predict",
        "training_method": trained["training_method"],
        "prediction_method": "rk4_high_precision",
        "ratio": trained["ratio"],
        "predict_ratio": predict_ratio,
        "seed": trained["seed"],
        "h_model": trained["h_model"],
        "h_predict": h,
        "noise_level": noise_level,
        "best_loss": trained["best_loss"],
        "train_mse": float(jnp.mean((pred_train - data["train_clean"]) ** 2)),
        "validation_mse": float(jnp.mean((pred_val - data["val_clean"]) ** 2)),
        "extrapolate_mse": float(
            jnp.mean(
                (
                    pred_extrap[:, extrap_start:, :]
                    - data["extrap"][:, extrap_start:, :]
                )
                ** 2
            )
        ),
        "instability_rate": instability_rate(pred_extrap),
        "final_mse": float(extrap_mse_by_time[-1]),
        "final_l2_error": float(
            jnp.linalg.norm(pred_extrap[:, -1, :] - data["extrap"][:, -1, :])
        ),
        **vector_field_metrics(params),
    }
    curves = {
        "train_mse_by_time": train_mse_by_time,
        "train_rel_error_by_time": train_rel_by_time,
        "train_mse_by_time_state": train_mse_by_time_state,
        "validation_mse_by_time": val_mse_by_time,
        "validation_rel_error_by_time": val_rel_by_time,
        "validation_mse_by_time_state": val_mse_by_time_state,
        "extrapolate_mse_by_time": extrap_mse_by_time,
        "extrapolate_rel_error_by_time": extrap_rel_by_time,
        "extrapolate_mse_by_time_state": extrap_mse_by_time_state,
    }
    return row, curves


def plot_error_vs_time(details, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, curves in details.items():
        seed, train_method, pred_method, predict_ratio = key
        label = (
            f"seed={seed}, "
            f"train={train_method}, "
            f"pred={pred_method}, "
            f"predict_ratio={predict_ratio}"
        )
        ax.semilogy(
            np.asarray(t_extrapolate),
            curves["extrapolate_mse_by_time"],
            label=label,
        )
    ax.axvline(T, linestyle=":", label="extrapolation starts")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE by time")
    ax.set_title("Diffrax Training + High-Precision RK4 Prediction")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_time_by_seed.png", dpi=200)
    plt.close(fig)


def run_diffrax_train_rk4_predict(
    output_dir="diffrax_train_rk4_predict_results",
    seeds=None,
    num_epochs=30000,
    step_size=3e-3,
    chunk_size=100,
    train_rtol=1e-6,
    train_atol=1e-6,
    predict_ratio=16,
):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(f"[setup] writing results to {output_dir}")
    progress("[setup] building clean reference data with Diffrax")
    data = make_reference_data()
    progress("[setup] reference data ready")

    trained_models = {}
    rows = []
    details = {}
    total_runs = len(seeds)

    for run_idx, seed in enumerate(seeds, start=1):
        progress(
            f"[train {run_idx}/{total_runs}] "
            f"start seed={seed}, method=diffrax_tsit5"
        )
        trained = train_model_diffrax(
            seed,
            data,
            num_epochs=num_epochs,
            step_size=step_size,
            chunk_size=chunk_size,
            rtol=train_rtol,
            atol=train_atol,
        )
        trained_models[(seed, "diffrax_tsit5", "adaptive")] = trained
        progress(
            f"[train {run_idx}/{total_runs}] "
            f"done seed={seed}, "
            f"best_loss={trained['best_loss']:.6e}, "
            f"best_epoch={trained['best_epoch']}"
        )

        progress(
            f"[eval {run_idx}/{total_runs}] "
            f"start seed={seed}, pred=rk4_high_precision, "
            f"predict_ratio={predict_ratio}"
        )
        row, curves = evaluate_with_high_precision_rk4(
            trained,
            data,
            predict_ratio=predict_ratio,
        )
        rows.append(row)
        key = (seed, "diffrax_tsit5", "rk4_high_precision", predict_ratio)
        details[key] = curves
        progress(
            f"[eval {run_idx}/{total_runs}] "
            f"done seed={seed}, "
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
                "trained": trained_models,
                "rows": rows,
                "details": details,
            },
            f,
        )

    progress("[plot] writing error_vs_time_by_seed.png")
    plot_error_vs_time(details, output_dir)
    progress("[done] Diffrax training + RK4 prediction experiment complete")
    return {
        "trained": trained_models,
        "rows": rows,
        "details": details,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with Diffrax/Tsit5 and evaluate with high-precision RK4."
    )
    parser.add_argument(
        "--output-dir",
        default="diffrax_train_rk4_predict_results",
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Initialization seeds.",
    )
    parser.add_argument("--num-epochs", type=int, default=30000)
    parser.add_argument("--step-size", type=float, default=3e-3)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--train-rtol", type=float, default=1e-6)
    parser.add_argument("--train-atol", type=float, default=1e-6)
    parser.add_argument(
        "--predict-ratio",
        type=int,
        default=16,
        help="High-precision RK4 substeps per observation interval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diffrax_train_rk4_predict(
        output_dir=args.output_dir,
        seeds=args.seeds,
        num_epochs=args.num_epochs,
        step_size=args.step_size,
        chunk_size=args.chunk_size,
        train_rtol=args.train_rtol,
        train_atol=args.train_atol,
        predict_ratio=args.predict_ratio,
    )
