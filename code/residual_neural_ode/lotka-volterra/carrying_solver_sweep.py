import argparse
import csv
import pickle
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import diffrax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax, random, value_and_grad, vmap

from solver import forward_euler, heun, rk4, roll_out


a, b, r, z, c = 1.0, 0.05, 1.5, 0.03, 0.001
noise_level = 0.01
T = 5
num_observed = 101
T_extrapolate = 4 * T
num_extrapolate_observed = 4 * (num_observed - 1) + 1
t_obs = jnp.linspace(0.0, T, num_observed)
t_extrapolate = jnp.linspace(0.0, T_extrapolate, num_extrapolate_observed)
obs_dt = float(t_obs[1] - t_obs[0])
state_scale = jnp.array([50.0, 50.0])

y0_batch = jnp.array(
    [
        [15.0, 25.0],
        [10.0, 20.0],
        [20.0, 30.0],
        [8.0, 18.0],
        [22.0, 26.0],
        [16.0, 15.0],
        [35.0, 20.0],
        [40.0, 35.0],
        [45.0, 15.0],
        [3.0, 20.0],
        [5.0, 10.0],
        [15.0, 45.0],
        [10.0, 50.0],
        [20.0, 5.0],
        [30.0, 8.0],
        [40.0, 40.0],
        [45.0, 5.0],
        [5.0, 45.0],
    ]
)

y0_validation = jnp.array(
    [
        [12.0, 35.0],
        [25.0, 12.0],
        [32.0, 28.0],
        [7.0, 35.0],
    ]
)

TRAINING_CONFIGS = [
    ("rk4_ratio4", rk4, 4),
    ("heun_ratio1", heun, 1),
    ("euler_ratio1", forward_euler, 1),
    ("euler_ratio16", forward_euler, 16),
    ("diffrax_tsit5", None, "adaptive"),
]


def progress(message):
    print(message, flush=True)


def lotka_volterra(t, y, args):
    prey, predator = y[0], y[1]
    return jnp.array(
        [
            a * prey - b * prey * predator - c * prey * prey,
            -r * predator + z * prey * predator,
        ]
    )


def true_rhs(y, params=None):
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
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
        max_steps=500000,
    )
    return sol.ys


def make_reference_data():
    train_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_batch)
    val_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_validation)
    extrap = vmap(lambda y0: solve_reference(y0, t_extrapolate))(y0_batch)
    key = random.key(42)
    train_noisy = train_clean + noise_level * train_clean * random.normal(
        key,
        train_clean.shape,
    )
    return {
        "train_clean": train_clean,
        "train_noisy": train_noisy,
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


def initial_params(seed):
    return {
        "nn_params": init_network_params([2, 64, 64, 2], random.key(seed)),
        "f_physics": jnp.array([0.7, 0.03, 1.0, 0.025]),
        "residual_scale": jnp.array(1.0),
    }


def make_optimizer(nn_lr, physics_lr):
    return optax.multi_transform(
        {
            "nn": optax.adam(nn_lr),
            "physics": optax.adam(physics_lr),
            "freeze": optax.set_to_zero(),
        },
        {
            "nn_params": "nn",
            "f_physics": "physics",
            "residual_scale": "freeze",
        },
    )


def nn(y, nn_parameters):
    activations = y / state_scale
    for weights, bias in nn_parameters[:-1]:
        activations = jnn.swish(jnp.dot(weights, activations) + bias)
    final_weights, final_bias = nn_parameters[-1]
    return jnp.dot(final_weights, activations) + final_bias


def f_physics(y, f_physics_params):
    prey, predator = y[0], y[1]
    return jnp.array(
        [
            f_physics_params[0] * prey - f_physics_params[1] * predator * prey,
            -f_physics_params[2] * predator + f_physics_params[3] * predator * prey,
        ]
    )


def model_rhs(y, params):
    return (
        f_physics(y, params["f_physics"])
        + params["residual_scale"] * nn(y, params["nn_params"])
    )


def solve_model_diffrax(y0, params, ts, t1):
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
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=500000,
    )
    return sol.ys


def solve_model_diffrax_batch(y0s, params, ts, t1):
    return vmap(lambda y0: solve_model_diffrax(y0, params, ts, t1))(y0s)


def l2_regularization(params, states):
    residuals = vmap(nn, in_axes=(0, None))(states, params["nn_params"])
    return jnp.mean(residuals**2)


def squared_correlation(u, v):
    u = u - jnp.mean(u)
    v = v - jnp.mean(v)
    return (jnp.sum(u * v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-8)) ** 2


def orthogonality_regularization(params, states):
    residuals = vmap(nn, in_axes=(0, None))(states, params["nn_params"])
    prey = states[:, 0]
    predator = states[:, 1]
    xy = prey * predator
    r1 = residuals[:, 0]
    r2 = residuals[:, 1]
    return (
        squared_correlation(r1, prey)
        + squared_correlation(r1, xy)
        + squared_correlation(r2, predator)
        + squared_correlation(r2, xy)
    )


def rollout_training_loss(params, y0, target, h, ratio, method):
    num_steps = (target.shape[0] - 1) * ratio
    pred = roll_out(y0, h, model_rhs, params, num_steps, method)[::ratio]
    return jnp.mean((target - pred) ** 2)


def diffrax_training_loss(params, y0, target):
    pred = solve_model_diffrax(y0, params, t_obs, T)
    return jnp.mean((target - pred) ** 2)


def make_batch_loss(method, ratio):
    if method is None:
        def batch_loss(params, y0s, targets):
            losses = vmap(diffrax_training_loss, in_axes=(None, 0, 0))(
                params,
                y0s,
                targets,
            )
            return jnp.mean(losses)

        return batch_loss

    h = obs_dt / ratio

    def sample_loss(params, y0, target):
        return rollout_training_loss(params, y0, target, h, ratio, method)

    def batch_loss(params, y0s, targets):
        losses = vmap(sample_loss, in_axes=(None, 0, 0))(
            params,
            y0s,
            targets,
        )
        return jnp.mean(losses)

    return batch_loss


def make_training_objective(batch_loss):
    def objective(params, y0s, targets, l2_weight, ortho_weight):
        data_loss = batch_loss(params, y0s, targets)
        states = targets.reshape(-1, 2)
        return (
            data_loss
            + l2_weight * l2_regularization(params, states)
            + ortho_weight * orthogonality_regularization(params, states)
        )

    return objective


def scaled_training_stages(stage_scale):
    stages = [
        {
            "name": "stage 1 | physics warm start",
            "epochs": 2000,
            "nn_lr": 0.0,
            "physics_lr": 3e-3,
            "residual_scale": 0.0,
            "l2_weight": 0.0,
            "ortho_weight": 0.0,
            "patience": None,
        },
        {
            "name": "stage 2 | open residual slowly",
            "epochs": 4000,
            "nn_lr": 1e-3,
            "physics_lr": 3e-4,
            "residual_scale": 1.0,
            "l2_weight": 0.0,
            "ortho_weight": 0.0,
            "patience": 1000,
        },
        {
            "name": "stage 3 | joint training",
            "epochs": 8000,
            "nn_lr": 1e-3,
            "physics_lr": 2e-4,
            "residual_scale": 1.0,
            "l2_weight": 0.0,
            "ortho_weight": 0.0,
            "patience": 1200,
        },
        {
            "name": "stage 4 | physics fine tune",
            "epochs": 1500,
            "nn_lr": 0.0,
            "physics_lr": 0.0,
            "residual_scale": 1.0,
            "l2_weight": 0.0,
            "ortho_weight": 0.0,
            "patience": 500,
        },
    ]
    if stage_scale == 1.0:
        return stages
    scaled = []
    for stage in stages:
        new_stage = dict(stage)
        new_stage["epochs"] = max(100, int(stage["epochs"] * stage_scale))
        if stage["patience"] is not None:
            new_stage["patience"] = max(100, int(stage["patience"] * stage_scale))
        scaled.append(new_stage)
    return scaled


def train_one(seed, config_name, method, ratio, data, chunk_size, stage_scale):
    params = initial_params(seed)
    batch_loss = make_batch_loss(method, ratio)
    training_objective = make_training_objective(batch_loss)

    def make_train_chunk(optimizer):
        @jit
        def train_chunk(params, opt_state, l2_weight, ortho_weight):
            def step(carry, _):
                params, opt_state = carry
                loss_val, grads = value_and_grad(training_objective)(
                    params,
                    y0_batch,
                    data["train_noisy"],
                    l2_weight,
                    ortho_weight,
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), loss_val

            (params, opt_state), losses = lax.scan(
                step,
                (params, opt_state),
                None,
                length=chunk_size,
            )
            return params, opt_state, losses

        return train_chunk

    best_loss = float("inf")
    best_params = params
    best_epoch = 0
    best_stage = ""
    min_delta = 1e-7
    global_epoch = 0
    stage_history = []

    for stage in scaled_training_stages(stage_scale):
        stage_counter = 0
        optimizer = make_optimizer(stage["nn_lr"], stage["physics_lr"])
        train_chunk = make_train_chunk(optimizer)
        params = {**params, "residual_scale": jnp.array(stage["residual_scale"])}
        opt_state = optimizer.init(params)
        l2_weight = jnp.array(stage["l2_weight"])
        ortho_weight = jnp.array(stage["ortho_weight"])
        n_chunks = stage["epochs"] // chunk_size
        progress(
            f"[train] {config_name}, seed={seed}, {stage['name']}, "
            f"chunks={n_chunks}"
        )

        for _ in range(n_chunks):
            params, opt_state, losses = train_chunk(
                params,
                opt_state,
                l2_weight,
                ortho_weight,
            )
            global_epoch += chunk_size
            objective_loss = float(losses[-1])
            train_data_loss = float(batch_loss(params, y0_batch, data["train_noisy"]))
            validation_loss = float(batch_loss(params, y0_validation, data["val_clean"]))
            stage_history.append(
                {
                    "stage": stage["name"],
                    "epoch": global_epoch,
                    "objective_loss": objective_loss,
                    "train_data_loss": train_data_loss,
                    "validation_loss": validation_loss,
                    "nn_lr": stage["nn_lr"],
                    "physics_lr": stage["physics_lr"],
                    "residual_scale": stage["residual_scale"],
                    "l2_weight": stage["l2_weight"],
                    "ortho_weight": stage["ortho_weight"],
                }
            )

            if best_loss - validation_loss > min_delta:
                best_loss = validation_loss
                best_params = params
                best_epoch = global_epoch
                best_stage = stage["name"]
                stage_counter = 0
            else:
                stage_counter += chunk_size

            if stage["patience"] is not None and stage_counter >= stage["patience"]:
                break

    return {
        "params": best_params,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "training_config": config_name,
        "training_method": method.__name__ if method is not None else "diffrax_tsit5",
        "ratio": ratio,
        "h_model": "adaptive" if method is None else obs_dt / ratio,
        "seed": seed,
        "stage_history": stage_history,
    }


def time_error_curves(pred, true):
    sq_err = (pred - true) ** 2
    mse_by_time = jnp.mean(sq_err, axis=(0, 2))
    rel_by_time = jnp.linalg.norm(pred - true, axis=(0, 2)) / (
        jnp.linalg.norm(true, axis=(0, 2)) + 1e-8
    )
    mse_by_time_state = jnp.mean(sq_err, axis=0)
    return np.asarray(mse_by_time), np.asarray(rel_by_time), np.asarray(mse_by_time_state)


def instability_rate(pred, threshold=1e6):
    pred = np.asarray(pred)
    bad = (
        ~np.isfinite(pred).all(axis=(1, 2))
        | (pred < 0).any(axis=(1, 2))
        | (np.abs(pred) > threshold).any(axis=(1, 2))
    )
    return float(np.mean(bad))


def vector_field_metrics(params):
    prey_vals = jnp.linspace(3.0, 50.0, 24)
    pred_vals = jnp.linspace(5.0, 55.0, 24)
    prey_grid, pred_grid = jnp.meshgrid(prey_vals, pred_vals)
    states = jnp.stack([prey_grid.reshape(-1), pred_grid.reshape(-1)], axis=1)
    true_vf = vmap(true_rhs, in_axes=(0, None))(states, None)
    physics_vf = vmap(f_physics, in_axes=(0, None))(states, params["f_physics"])
    model_vf = vmap(model_rhs, in_axes=(0, None))(states, params)
    nn_vf = vmap(lambda s: params["residual_scale"] * nn(s, params["nn_params"]))(states)
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


def evaluate(trained, data):
    params = trained["params"]
    pred_train = solve_model_diffrax_batch(y0_batch, params, t_obs, T)
    pred_val = solve_model_diffrax_batch(y0_validation, params, t_obs, T)
    pred_extrap = solve_model_diffrax_batch(
        y0_batch,
        params,
        t_extrapolate,
        T_extrapolate,
    )
    train_mse_by_time, train_rel_by_time, train_mse_by_time_state = time_error_curves(
        pred_train,
        data["train_clean"],
    )
    val_mse_by_time, val_rel_by_time, val_mse_by_time_state = time_error_curves(
        pred_val,
        data["val_clean"],
    )
    extrap_mse_by_time, extrap_rel_by_time, extrap_mse_by_time_state = time_error_curves(
        pred_extrap,
        data["extrap"],
    )
    extrap_start = num_observed
    row = {
        "experiment_kind": "carrying_solver_sweep",
        "training_config": trained["training_config"],
        "training_method": trained["training_method"],
        "ratio": trained["ratio"],
        "seed": trained["seed"],
        "h_model": trained["h_model"],
        "evaluation_method": "diffrax_tsit5",
        "noise_level": noise_level,
        "best_loss": trained["best_loss"],
        "best_epoch": trained["best_epoch"],
        "best_stage": trained["best_stage"],
        "train_mse": float(jnp.mean((pred_train - data["train_clean"]) ** 2)),
        "validation_mse": float(jnp.mean((pred_val - data["val_clean"]) ** 2)),
        "extrapolate_mse": float(
            jnp.mean((pred_extrap[:, extrap_start:, :] - data["extrap"][:, extrap_start:, :]) ** 2)
        ),
        "final_mse": float(extrap_mse_by_time[-1]),
        "final_l2_error": float(
            jnp.linalg.norm(pred_extrap[:, -1, :] - data["extrap"][:, -1, :])
        ),
        "instability_rate": instability_rate(pred_extrap),
        "learned_f_physics_a": float(params["f_physics"][0]),
        "learned_f_physics_b": float(params["f_physics"][1]),
        "learned_f_physics_r": float(params["f_physics"][2]),
        "learned_f_physics_z": float(params["f_physics"][3]),
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


def write_summary_csv(rows, path):
    fieldnames = [
        "experiment_kind",
        "training_config",
        "training_method",
        "ratio",
        "seed",
        "h_model",
        "evaluation_method",
        "noise_level",
        "best_loss",
        "best_epoch",
        "best_stage",
        "train_mse",
        "validation_mse",
        "extrapolate_mse",
        "final_mse",
        "final_l2_error",
        "instability_rate",
        "residual_mse",
        "residual_rel_error",
        "model_vf_mse",
        "model_vf_rel_error",
        "learned_f_physics_a",
        "learned_f_physics_b",
        "learned_f_physics_r",
        "learned_f_physics_z",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(output_dir):
    summary_path = output_dir / "carrying_solver_sweep_summary.csv"
    details_path = output_dir / "carrying_solver_sweep_details.pkl"
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
        (row.get("training_config"), int(row.get("seed")))
        for row in rows
        if row.get("training_config") and row.get("seed") not in (None, "")
    }
    return rows, details, trained_models, completed


def save_results(output_dir, rows, details, trained_models):
    write_summary_csv(rows, output_dir / "carrying_solver_sweep_summary.csv")
    with (output_dir / "carrying_solver_sweep_details.pkl").open("wb") as f:
        pickle.dump(
            {
                "trained": trained_models,
                "rows": rows,
                "details": details,
                "training_configs": [
                    (name, method.__name__ if method is not None else "diffrax_tsit5", ratio)
                    for name, method, ratio in TRAINING_CONFIGS
                ],
            },
            f,
        )


def run_sweep(output_dir, seeds, chunk_size, stage_scale):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[setup] writing results to {output_dir}")
    progress("[setup] building reference data")
    data = make_reference_data()
    progress("[setup] reference data ready")

    rows, details, trained_models, completed = load_existing_results(output_dir)
    if completed:
        progress(f"[resume] found {len(completed)} completed runs; skipping them")
    total = len(TRAINING_CONFIGS) * len(seeds)
    run_idx = 0
    for config_name, method, ratio in TRAINING_CONFIGS:
        for seed in seeds:
            run_idx += 1
            if (config_name, seed) in completed:
                progress(f"[run {run_idx}/{total}] skip completed {config_name}, seed={seed}")
                continue
            progress(f"[run {run_idx}/{total}] train {config_name}, seed={seed}")
            trained = train_one(
                seed,
                config_name,
                method,
                ratio,
                data,
                chunk_size=chunk_size,
                stage_scale=stage_scale,
            )
            trained_models[(config_name, seed)] = trained
            progress(
                f"[run {run_idx}/{total}] eval {config_name}, seed={seed}, "
                f"best_loss={trained['best_loss']:.6e}"
            )
            row, curves = evaluate(trained, data)
            rows.append(row)
            details[(config_name, seed)] = curves
            progress(
                f"[run {run_idx}/{total}] done {config_name}, seed={seed}, "
                f"val={row['validation_mse']:.6e}, extrap={row['extrapolate_mse']:.6e}, "
                f"residual={row['residual_mse']:.6e}"
            )
            save_results(output_dir, rows, details, trained_models)

    save_results(output_dir, rows, details, trained_models)
    progress("[done] carrying solver sweep complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep selected training solvers for carrying-capacity Lotka-Volterra."
    )
    parser.add_argument(
        "--output-dir",
        default="carrying_solver_sweep_results",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
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
    run_sweep(
        output_dir=args.output_dir,
        seeds=args.seeds,
        chunk_size=args.chunk_size,
        stage_scale=args.stage_scale,
    )
