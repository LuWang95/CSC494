import csv
import pickle
from pathlib import Path
from functools import partial

from jax import config

config.update("jax_enable_x64", True)

import diffrax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import jit, lax, random, value_and_grad, vmap

from solver import forward_euler, heun, rk4, roll_out

solvers = [forward_euler, heun, rk4]
a, b, r, z, c = 1.0, 0.05, 1.5, 0.03, 0.005
noise_level = 0
T = 5
num_observed = 101
T_extrapolate = 4 * T
num_extrapolate_observed = 4 * (num_observed - 1) + 1
t_obs = jnp.linspace(0.0, T, num_observed)
t_extrapolate = jnp.linspace(0.0, T_extrapolate, num_extrapolate_observed)
obs_dt = float(t_obs[1] - t_obs[0])


def progress(message):
    print(message, flush=True)


y0_batch = jnp.array(
    [
        [15.0, 25.0],
        [10.0, 20.0],
        [20.0, 30.0],
        [12.0, 22.0],
        [18.0, 28.0],
        [8.0, 18.0],
        [22.0, 26.0],
        [16.0, 15.0],
    ]
)
y0_validation = jnp.array([[17.0, 25.0], [10.0, 23.0], [19.0, 24.0]])


def lotka_volterra(t, y, args):
    prey, predator = y[0], y[1]
    return jnp.array(
        [
            a * prey - b * prey * predator - c * prey * prey,
            -r * predator + z * prey * predator,
        ]
    )


def true_rhs(y, params):
    return lotka_volterra(0.0, y, None)


def solve_reference(y0, ts):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lotka_volterra),
        diffrax.Tsit5(),
        t0=0.0,
        t1=float(ts[-1]),
        dt0=1e-3,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
        max_steps=500000,
    )
    return sol.ys


def make_reference_data():
    train_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_batch)
    val_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_validation)
    extrap = vmap(lambda y0: solve_reference(y0, t_extrapolate))(y0_batch)
    key = random.key(42)
    noisy_train = train_clean + noise_level * train_clean * random.normal(
        key, train_clean.shape
    )
    return {
        "train_clean": train_clean,
        "train_noisy": noisy_train,
        "val_clean": val_clean,
        "extrap": extrap,
    }


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes) - 1)
    return [
        random_layer_params(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


state_scale = jnp.array([50.0, 50.0])


def nn(y, nn_params):
    activations = y / state_scale
    for w, bias in nn_params[:-1]:
        activations = jnn.swish(jnp.dot(w, activations) + bias)
    final_w, final_bias = nn_params[-1]
    return jnp.dot(final_w, activations) + final_bias


def f_physics(y, f_physics_params):
    prey, predator = y[0], y[1]
    return jnp.array(
        [
            f_physics_params[0] * prey - f_physics_params[1] * predator * prey,
            -f_physics_params[2] * predator + f_physics_params[3] * predator * prey,
        ]
    )


def model_rhs(y, params):
    return f_physics(y, params["f_physics"]) + nn(y, params["nn_params"])


def initial_params(seed):
    return {
        "nn_params": init_network_params([2, 64, 64, 2], random.key(seed)),
        "f_physics": jnp.array([1.0, 0.05, 1.5, 0.03]),
    }


def make_optimizer(step_size):
    return optax.multi_transform(
        {"train": optax.adam(step_size), "freeze": optax.set_to_zero()},
        {"nn_params": "train", "f_physics": "freeze"},
    )


def train_model(seed, method, ratio, data, num_epochs, step_size, chunk_size):
    h = obs_dt / ratio
    num_steps = int((num_observed - 1) * ratio)
    params = initial_params(seed)
    optimizer = make_optimizer(step_size)
    opt_state = optimizer.init(params)

    def sample_loss(parameters, y0, target):
        pred = roll_out(y0, h, model_rhs, parameters, num_steps, method)[::ratio]
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
            train_step, (parameters, opt_state), None, length=chunk_size
        )
        return parameters, opt_state, losses

    best_loss = float("inf")
    best_params = params
    patience = 500
    counter = 0
    for _ in range(num_epochs // chunk_size):
        params, opt_state, losses = train_chunk(params, opt_state)
        loss_val = float(losses[-1])
        if best_loss - loss_val > 1e-7:
            best_loss = loss_val
            best_params = params
            counter = 0
        else:
            counter += chunk_size
        if counter >= patience:
            break
    return {
        "params": best_params,
        "best_loss": best_loss,
        "training_method": method.__name__,
        "ratio": ratio,
        "h_model": h,
        "seed": seed,
    }


def rollout_batch(y0s, h, rhs, params, num_steps, method):
    return vmap(lambda y0: roll_out(y0, h, rhs, params, num_steps, method))(y0s)


def solve_model_diffrax(y0, params, ts):
    def rhs(t, y, args):
        return model_rhs(y, args)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=float(ts[-1]),
        dt0=1e-3,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-9),
        max_steps=500000,
    )
    return sol.ys


def solve_model_diffrax_batch(y0s, params, ts):
    return vmap(lambda y0: solve_model_diffrax(y0, params, ts))(y0s)


def time_error_curves(pred, true):
    sq_err = (pred - true) ** 2
    # overall: average over batch and state
    mse_by_time = jnp.mean(sq_err, axis=(0, 2))
    # per-state: average only over batch
    mse_by_time_state = jnp.mean(sq_err, axis=0)
    rel_by_time = jnp.linalg.norm(pred - true, axis=(0, 2)) / (jnp.linalg.norm(true, axis=(0, 2)) + 1e-8)
    return (
        np.asarray(mse_by_time),
        np.asarray(rel_by_time),
        np.asarray(mse_by_time_state),
    )


def instability_rate(pred, blowup_threshold=1e6):
    pred = np.asarray(pred)
    bad = (~np.isfinite(pred).all(axis=(1, 2)) | (pred < 0).any(axis=(1, 2)) | (np.abs(pred) > blowup_threshold).any(axis=(1, 2)))
    return float(np.mean(bad))


def vector_field_metrics(params):
    prey_vals = jnp.linspace(5.0, 25.0, 20)
    pred_vals = jnp.linspace(10.0, 40.0, 20)
    prey_grid, pred_grid = jnp.meshgrid(prey_vals, pred_vals)
    states = jnp.stack([prey_grid.reshape(-1), pred_grid.reshape(-1)], axis=1)
    true_vf = vmap(true_rhs, in_axes=(0, None))(states, None)
    physics_vf = vmap(f_physics, in_axes=(0, None))(states, params["f_physics"])
    model_vf = vmap(model_rhs, in_axes=(0, None))(states, params)
    nn_vf = vmap(nn, in_axes=(0, None))(states, params["nn_params"])
    true_residual = true_vf - physics_vf
    return {
        "residual_mse": float(jnp.mean((nn_vf - true_residual) ** 2)),
        "residual_rel_error": float(
            jnp.linalg.norm(nn_vf - true_residual)
            / (jnp.linalg.norm(true_residual) + 1e-8)
        ),
        "model_vf_mse": float(jnp.mean((model_vf - true_vf) ** 2)),
        "model_vf_rel_error": float(
            jnp.linalg.norm(model_vf - true_vf) / (jnp.linalg.norm(true_vf) + 1e-8)
        ),
    }


def evaluate_true_rhs(method, ratio, data):
    h = obs_dt / ratio
    steps = int((num_extrapolate_observed - 1) * ratio)
    pred = rollout_batch(y0_batch, h, true_rhs, None, steps, method)[:, ::ratio, :]
    true = data["extrap"]
    mse_by_time, rel_by_time, mse_by_time_state = time_error_curves(pred, true)
    extrap_start = num_observed
    row = {
        "experiment_kind": "true_rhs_baseline",
        "training_method": "none",
        "prediction_method": method.__name__,
        "ratio": "N/A",
        "predict_ratio": ratio,
        "h_model": "N/A",
        "h_predict": h,
        "noise_level": 0.0,
        "best_loss": "N/A",
        "train_mse": float(jnp.mean((pred[:, :num_observed, :] - true[:, :num_observed, :]) ** 2)),
        "validation_mse": "N/A",
        "extrapolate_mse": float(
            jnp.mean((pred[:, extrap_start:, :] - true[:, extrap_start:, :]) ** 2)
        ),
        "residual_mse": "N/A",
        "residual_rel_error": "N/A",
        "model_vf_mse": "N/A",
        "model_vf_rel_error": "N/A",
        "instability_rate": instability_rate(pred),
        "final_mse": float(mse_by_time[-1]),
        "final_l2_error": float(jnp.linalg.norm(pred[:, -1, :] - true[:, -1, :])),
    }
    return row, {
        "extrapolate_mse_by_time": mse_by_time,
        "extrapolate_rel_error_by_time": rel_by_time,
        "extrapolate_mse_by_time_state": mse_by_time_state,
    }


def evaluate_model_diffrax(trained, data, experiment_type):
    params = trained["params"]
    pred_train = solve_model_diffrax_batch(y0_batch, params, t_obs)
    pred_val = solve_model_diffrax_batch(y0_validation, params, t_obs)
    pred_extrap = solve_model_diffrax_batch(y0_batch, params, t_extrapolate)
    train_mse_by_time, train_rel_by_time, train_mse_by_time_state = time_error_curves(
        pred_train, data["train_clean"]
    )
    val_mse_by_time, val_rel_by_time,val_mse_by_time_state = time_error_curves(pred_val, data["val_clean"])
    extrap_mse_by_time, extrap_rel_by_time, extrap_mse_by_time_state = time_error_curves(
        pred_extrap, data["extrap"]
    )
    extrap_start = num_observed
    row = {
        "experiment_kind": experiment_type,
        "training_method": trained["training_method"],
        "prediction_method": "diffrax_tsit5",
        "ratio": trained["ratio"],
        "predict_ratio": "adaptive",
        "seed": trained["seed"],
        "h_model": trained["h_model"],
        "h_predict": "adaptive",
        "noise_level": noise_level,
        "best_loss": trained["best_loss"],
        "train_mse": float(jnp.mean((pred_train - data["train_clean"]) ** 2)),
        "validation_mse": float(jnp.mean((pred_val - data["val_clean"]) ** 2)),
        "extrapolate_mse": float(
            jnp.mean((pred_extrap[:, extrap_start:, :] - data["extrap"][:, extrap_start:, :]) ** 2)
        ),
        "instability_rate": instability_rate(pred_extrap),
        "final_mse": float(extrap_mse_by_time[-1]),
        "final_l2_error": float(
            jnp.linalg.norm(pred_extrap[:, -1, :] - data["extrap"][:, -1, :])
        ),
        **vector_field_metrics(params),
    }
    return row, {
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


def write_summary_csv(rows, path):
    fieldnames = [
        "experiment_kind",
        "training_method",
        "prediction_method",
        "ratio",
        "predict_ratio",
        "seed",
        "h_model",
        "h_predict",
        "noise_level",
        "best_loss",
        "train_mse",
        "validation_mse",
        "extrapolate_mse",
        "residual_mse",
        "residual_rel_error",
        "model_vf_mse",
        "model_vf_rel_error",
        "instability_rate",
        "final_mse",
        "final_l2_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_error_vs_stepsize(rows, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for methods in solvers:
        method_rows = [
            row
            for row in rows
            if row["experiment_kind"] == "true_rhs_baseline"
            and row["prediction_method"] == methods.__name__
        ]
        method_rows.sort(key=lambda row: float(row["h_predict"]))
        hs = np.array([float(row["h_predict"]) for row in method_rows])
        errors = np.array([float(row["final_l2_error"]) for row in method_rows])
        ax.loglog(hs, errors, marker="o", label=methods.__name__)
        if len(hs) >= 2 and np.all(errors > 0):
            slope = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            ax.text(hs[-1], errors[-1], f"slope={slope:.2f}", fontsize=8)
    ax.set_xlabel("step size")
    ax.set_ylabel("final L2 error")
    ax.set_title("True RHS numerical order check")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_stepsize_loglog.png", dpi=200)


def plot_error_vs_time(details, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, curves in details.items():
        if len(key) == 5:
            seed, train_method_name, train_ratio, pred_method_name, predict_ratio = key
            seed_label = f", seed={seed}"
        else:
            train_method_name, train_ratio, pred_method_name, predict_ratio = key
            seed_label = ""
        label = (
            f"train={train_method_name}, "
            f"train_ratio={train_ratio}, "
            f"pred={pred_method_name}, "
            f"predict_ratio={predict_ratio}"
            f"{seed_label}"
        )
        ax.semilogy(
            np.asarray(t_extrapolate),
            curves["extrapolate_mse_by_time"],
            label=label,
        )
    ax.axvline(T, linestyle=":", label="extrapolation starts")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE by time")
    ax.set_title("Prediction solver error propagation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_time_by_solver.png", dpi=200)

def run_rhs_baseline(output_dir="rhs_baseline_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = make_reference_data()
    rows = []
    details = {}

    ratios = [1, 2, 4, 8]

    for method in solvers:
        for ratio in ratios:
            row, curves = evaluate_true_rhs(method, ratio, data)
            rows.append(row)
            details[(method.__name__,ratio)] = curves
            print(
                method.__name__,
                "ratio =", ratio,
                "h =", row["h_predict"],
                "final_mse =", row["final_mse"],
                "final_l2_error =",
                row["final_l2_error"],
                "instability =", row["instability_rate"],
            )

    write_summary_csv(rows, output_dir / "rhs_baseline_summary.csv")

    with (output_dir / "rhs_baseline_details.pkl").open("wb") as f:
        pickle.dump({"rows": rows, "details": details}, f)

    plot_error_vs_stepsize(rows, output_dir)


def run_prediction_sweep(output_dir="no_noise_diffrax_prediction_sweep_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(f"[setup] writing results to {output_dir}")
    progress("[setup] building clean reference data with Diffrax")
    data = make_reference_data()
    progress("[setup] reference data ready")

    all_trained = {}
    seeds = [0, 1, 2, 3, 4]
    train_methods = [heun, rk4]
    train_ratios = [1, 2, 4, 8, 16]
    total_runs = len(seeds) * len(train_methods) * len(train_ratios)
    run_idx = 0

    for train_method in train_methods:
        for train_ratio in train_ratios:
            for seed in seeds:
                run_idx += 1
                progress(
                    f"[train {run_idx}/{total_runs}] "
                    f"start seed={seed}, "
                    f"method={train_method.__name__}, "
                    f"train_ratio={train_ratio}"
                )
                trained = train_model(
                    seed,
                    train_method,
                    train_ratio,
                    data,
                    num_epochs=30000,
                    step_size=3e-3,
                    chunk_size=100,
                )

                all_trained[(seed,train_method.__name__, train_ratio)] = trained
                progress(
                    f"[train {run_idx}/{total_runs}] "
                    f"done seed={seed}, "
                    f"method={train_method.__name__}, "
                    f"train_ratio={train_ratio}, "
                    f"best_loss={trained['best_loss']:.6e}"
                )

    rows = []
    details = {}
    progress("[eval] starting Diffrax/Tsit5 evaluation")

    for eval_idx, ((seed, train_method_name, train_ratio), trained) in enumerate(
        all_trained.items(),
        start=1,
    ):
        progress(
            f"[eval {eval_idx}/{total_runs}] "
            f"start seed={seed}, "
            f"method={train_method_name}, "
            f"train_ratio={train_ratio}"
        )
        row, curves = evaluate_model_diffrax(
            trained=trained,
            data=data,
            experiment_type="prediction_sweep_diffrax_eval",
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
            f"done seed={seed}, "
            f"method={train_method_name}, "
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
    progress("[done] no-noise Diffrax prediction sweep complete")
    return {
        "trained": all_trained,
        "rows": rows,
        "details": details,
    }


if __name__ == "__main__":
    run_prediction_sweep()