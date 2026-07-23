import argparse
import csv
import pickle
from pathlib import Path

from carrying_solver_sweep import (
    initial_params,
    instability_rate,
    l2_regularization,
    make_optimizer,
    model_rhs,
    orthogonality_regularization,
    progress,
    scaled_training_stages,
    solve_model_diffrax_batch,
    solve_reference,
    time_error_curves,
    true_rhs,
    vector_field_metrics,
    y0_batch,
    y0_validation,
)
from jax import jit, lax, random, value_and_grad, vmap
import jax.numpy as jnp
import numpy as np
import optax
from solver import rk4, roll_out
from experiment_metadata import write_run_manifest


OBS_DT = 0.05
T_FINAL = 20.0
DEFAULT_TRAIN_INTERVALS = [1.25, 2.5, 5.0, 7.5]
DEFAULT_RATIO = 8
DEFAULT_NOISE_LEVEL = 0.0
DEFAULT_COVERAGE_RADIUS = 2.5
COVERAGE_FRACTION_HIGH = 0.8
COVERAGE_FRACTION_LOW = 0.2
FIXED_EXTRAP_HORIZONS = [2.5, 5.0, 10.0]
COVERAGE_EVAL_HORIZON = 10.0
EVALUATION_SCHEMA_VERSION = 3
TRAINING_PROTOCOL_VERSION = 2
TRAINING_METHOD = rk4
TRAINING_METHOD_NAME = "rk4"

REGULARIZATION_PROFILE = {
    "name": "none",
    "l2_weight": 0.0,
    "ortho_weight": 0.0,
}

Y0_COVERAGE_CANDIDATES = jnp.array(
    [
        [12.0, 35.0],
        [25.0, 12.0],
        [32.0, 28.0],
        [7.0, 35.0],
        [18.0, 22.0],
        [14.0, 30.0],
        [28.0, 18.0],
        [10.0, 40.0],
        [22.0, 38.0],
        [16.0, 12.0],
        [30.0, 32.0],
        [8.0, 28.0],
        [2.0, 8.0],
        [48.0, 48.0],
        [6.0, 50.0],
        [50.0, 8.0],
        [4.0, 45.0],
        [45.0, 4.0],
        [3.0, 15.0],
        [42.0, 42.0],
    ]
)


def parse_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    value = str(value).strip()
    if value == "" or value.upper() in {"N/A", "NULL", "NONE", "NAN"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=None):
    number = parse_float(value, default=np.nan)
    if not np.isfinite(number):
        return default
    return int(number)


def interval_observation_count(T_train):
    return int(round(T_train / OBS_DT)) + 1


def final_observation_count():
    return int(round(T_FINAL / OBS_DT)) + 1


def validate_sweep_args(
    train_intervals,
    ratio,
    noise_level,
    coverage_radius,
    chunk_size,
    stage_scale,
):
    if not train_intervals:
        raise ValueError("At least one training interval is required.")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    if noise_level < 0.0:
        raise ValueError("noise_level must be non-negative.")
    if coverage_radius <= 0.0:
        raise ValueError("coverage_radius must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if stage_scale <= 0.0:
        raise ValueError("stage_scale must be positive.")

    max_horizon = max(FIXED_EXTRAP_HORIZONS)
    for train_interval in train_intervals:
        if not np.isfinite(train_interval) or train_interval <= 0.0:
            raise ValueError("Every training interval must be finite and positive.")
        grid_steps = train_interval / OBS_DT
        if not np.isclose(grid_steps, round(grid_steps), atol=1e-9, rtol=0.0):
            raise ValueError(
                f"T_train={train_interval:g} is not aligned with OBS_DT={OBS_DT:g}."
            )
        if train_interval + max_horizon > T_FINAL + 1e-9:
            raise ValueError(
                f"T_train={train_interval:g} cannot support fixed H={max_horizon:g} "
                f"with T_FINAL={T_FINAL:g}."
            )


def build_config_name(train_interval, ratio):
    return f"{TRAINING_METHOD_NAME}_ratio{ratio}_T{train_interval:g}"


def run_key(
    train_interval,
    ratio,
    noise_level,
    coverage_radius,
    seed,
    chunk_size,
    stage_scale,
    training_protocol_version=TRAINING_PROTOCOL_VERSION,
):
    return (
        float(train_interval),
        int(ratio),
        float(noise_level),
        float(coverage_radius),
        int(seed),
        int(chunk_size),
        float(stage_scale),
        int(training_protocol_version),
    )


def row_run_key(row):
    return run_key(
        parse_float(row.get("train_interval")),
        parse_int(row.get("ratio"), default=DEFAULT_RATIO),
        parse_float(row.get("noise_level"), default=DEFAULT_NOISE_LEVEL),
        parse_float(row.get("coverage_radius"), default=DEFAULT_COVERAGE_RADIUS),
        parse_int(row.get("seed")),
        parse_int(row.get("chunk_size")),
        parse_float(row.get("stage_scale")),
        parse_int(
            row.get("training_protocol_version"),
            default=TRAINING_PROTOCOL_VERSION,
        ),
    )


def horizon_field_name(horizon):
    label = f"{horizon:g}".replace(".", "p")
    return f"same_ic_extrap_mse_h{label}"


def trajectory_coverage_stats(trajectory, state_cloud, radius):
    distances = jnp.linalg.norm(
        trajectory[:, None, :] - state_cloud[None, :, :],
        axis=2,
    )
    min_distances = jnp.min(distances, axis=1)
    covered_mask = min_distances <= radius
    return {
        "coverage_fraction": float(jnp.mean(covered_mask)),
        "mean_min_distance": float(jnp.mean(min_distances)),
        "max_min_distance": float(jnp.max(min_distances)),
    }


def trajectory_coverage_stats_with_extrap_split(trajectory, state_cloud, radius, extrap_start):
    full_stats = trajectory_coverage_stats(trajectory, state_cloud, radius)
    if extrap_start < trajectory.shape[0]:
        extrap_stats = trajectory_coverage_stats(trajectory[extrap_start:], state_cloud, radius)
        full_stats["extrap_coverage_fraction"] = extrap_stats["coverage_fraction"]
    else:
        full_stats["extrap_coverage_fraction"] = float("nan")
    return full_stats


def classify_candidate_trajectories(state_cloud, coverage_radius, t_final_grid, extrap_start):
    covered_y0 = []
    partial_y0 = []
    uncovered_y0 = []

    coverage_fractions = []
    coverage_groups = []
    coverage_end = min(
        t_final_grid.shape[0],
        extrap_start + int(round(COVERAGE_EVAL_HORIZON / OBS_DT)),
    )

    for y0 in Y0_COVERAGE_CANDIDATES:
        trajectory = solve_reference(y0, t_final_grid)
        stats = trajectory_coverage_stats_with_extrap_split(
            trajectory[:coverage_end],
            state_cloud,
            coverage_radius,
            extrap_start,
        )
        fraction = stats["extrap_coverage_fraction"]
        coverage_fractions.append(fraction)
        if fraction >= COVERAGE_FRACTION_HIGH:
            covered_y0.append(y0)
            coverage_groups.append("covered")
        elif fraction <= COVERAGE_FRACTION_LOW:
            uncovered_y0.append(y0)
            coverage_groups.append("uncovered")
        else:
            partial_y0.append(y0)
            coverage_groups.append("partial")

    return (
        covered_y0,
        partial_y0,
        uncovered_y0,
        jnp.asarray(coverage_fractions),
        coverage_groups,
    )


def stack_reference_trajectories(y0_list, t_final_grid):
    if not y0_list:
        num_final = t_final_grid.shape[0]
        return jnp.zeros((0, num_final, 2))
    return vmap(lambda y0: solve_reference(y0, t_final_grid))(jnp.stack(y0_list))


def horizon_mse(pred, true, start_idx, horizon, obs_dt):
    if pred.shape[0] == 0:
        return float("nan")
    # start_idx is already the first point strictly after T_train.  Therefore
    # H / dt samples end exactly at T_train + H; adding one would include H+dt.
    end_idx = start_idx + int(round(horizon / obs_dt))
    if end_idx > pred.shape[1]:
        raise ValueError(f"Fixed horizon H={horizon:g} exceeds the evaluation grid.")
    if end_idx <= start_idx:
        return float("nan")
    return float(
        jnp.mean((pred[:, start_idx:end_idx, :] - true[:, start_idx:end_idx, :]) ** 2)
    )


def candidate_horizon_mse(pred, true, start_idx, horizon, obs_dt):
    end_idx = start_idx + int(round(horizon / obs_dt))
    if end_idx > pred.shape[1]:
        raise ValueError(f"Fixed horizon H={horizon:g} exceeds the evaluation grid.")
    if end_idx <= start_idx:
        return jnp.full((pred.shape[0],), jnp.nan)
    return jnp.mean(
        (pred[:, start_idx:end_idx, :] - true[:, start_idx:end_idx, :]) ** 2,
        axis=(1, 2),
    )


def subset_vf_error(model_vf, true_vf, mask):
    if not bool(jnp.any(mask)):
        return float("nan"), float("nan")

    diff = model_vf[mask] - true_vf[mask]
    mse = float(jnp.mean(diff**2))
    rel = float(
        jnp.linalg.norm(diff)
        / (jnp.linalg.norm(true_vf[mask]) + 1e-8)
    )
    return mse, rel


def vector_field_metrics_with_coverage(params, state_cloud, coverage_radius):
    metrics = dict(vector_field_metrics(params))

    prey_vals = jnp.linspace(3.0, 50.0, 24)
    pred_vals = jnp.linspace(5.0, 55.0, 24)
    prey_grid, pred_grid = jnp.meshgrid(prey_vals, pred_vals)
    states = jnp.stack([prey_grid.reshape(-1), pred_grid.reshape(-1)], axis=1)
    true_vf = vmap(true_rhs, in_axes=(0, None))(states, None)
    model_vf = vmap(model_rhs, in_axes=(0, None))(states, params)

    cloud_dist = jnp.min(
        jnp.linalg.norm(states[:, None, :] - state_cloud[None, :, :], axis=2),
        axis=1,
    )
    seen_mask = cloud_dist <= coverage_radius
    unseen_mask = ~seen_mask

    seen_mse, seen_rel = subset_vf_error(model_vf, true_vf, seen_mask)
    unseen_mse, unseen_rel = subset_vf_error(model_vf, true_vf, unseen_mask)

    true_vf_train = vmap(true_rhs, in_axes=(0, None))(state_cloud, None)
    model_vf_train = vmap(model_rhs, in_axes=(0, None))(state_cloud, params)
    train_diff = model_vf_train - true_vf_train
    train_mse = float(jnp.mean(train_diff**2))
    train_rel = float(
        jnp.linalg.norm(train_diff) / (jnp.linalg.norm(true_vf_train) + 1e-8)
    )

    metrics.update(
        {
            "model_vf_mse_seen": seen_mse,
            "model_vf_rel_error_seen": seen_rel,
            "model_vf_mse_unseen": unseen_mse,
            "model_vf_rel_error_unseen": unseen_rel,
            "model_vf_mse_training_states": train_mse,
            "model_vf_rel_error_training_states": train_rel,
            "vf_grid_n_seen": int(jnp.sum(seen_mask)),
            "vf_grid_n_unseen": int(jnp.sum(unseen_mask)),
        }
    )
    return metrics


def make_batch_loss(method, ratio, obs_dt):
    h = obs_dt / ratio

    def sample_loss(params, y0, target):
        num_steps = (target.shape[0] - 1) * ratio
        pred = roll_out(y0, h, model_rhs, params, num_steps, method)[::ratio]
        return jnp.mean((target - pred) ** 2)

    def batch_loss(params, y0s, targets):
        losses = vmap(sample_loss, in_axes=(None, 0, 0))(params, y0s, targets)
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


def make_reference_data_for_interval(T_train, noise_level, coverage_radius):
    num_observed = interval_observation_count(T_train)
    num_final = final_observation_count()
    extrap_start = num_observed
    t_obs = jnp.linspace(0.0, T_train, num_observed)
    t_final_grid = jnp.linspace(0.0, T_FINAL, num_final)

    train_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_batch)
    val_clean = vmap(lambda y0: solve_reference(y0, t_obs))(y0_validation)
    extrap = vmap(lambda y0: solve_reference(y0, t_final_grid))(y0_batch)

    if noise_level > 0.0:
        key = random.key(42)
        train_noisy = train_clean + noise_level * train_clean * random.normal(
            key,
            train_clean.shape,
        )
    else:
        train_noisy = train_clean

    state_cloud = train_clean.reshape(-1, 2)
    (
        covered_y0,
        partial_y0,
        uncovered_y0,
        candidate_coverage_fractions,
        candidate_coverage_groups,
    ) = classify_candidate_trajectories(
        state_cloud,
        coverage_radius,
        t_final_grid,
        extrap_start,
    )

    y0_covered = jnp.stack(covered_y0) if covered_y0 else jnp.zeros((0, 2))
    y0_partial = jnp.stack(partial_y0) if partial_y0 else jnp.zeros((0, 2))
    y0_uncovered = jnp.stack(uncovered_y0) if uncovered_y0 else jnp.zeros((0, 2))
    covered_clean = stack_reference_trajectories(covered_y0, t_final_grid)
    partial_clean = stack_reference_trajectories(partial_y0, t_final_grid)
    uncovered_clean = stack_reference_trajectories(uncovered_y0, t_final_grid)
    candidate_clean = vmap(lambda y0: solve_reference(y0, t_final_grid))(
        Y0_COVERAGE_CANDIDATES
    )

    return {
        "T_train": T_train,
        "obs_dt": OBS_DT,
        "num_observed": num_observed,
        "extrap_start": extrap_start,
        "t_obs": t_obs,
        "T_final": T_FINAL,
        "t_final_grid": t_final_grid,
        "train_clean": train_clean,
        "train_noisy": train_noisy,
        "val_clean": val_clean,
        "extrap": extrap,
        "state_cloud": state_cloud,
        "y0_covered": y0_covered,
        "y0_partial": y0_partial,
        "y0_uncovered": y0_uncovered,
        "covered_clean": covered_clean,
        "partial_clean": partial_clean,
        "uncovered_clean": uncovered_clean,
        "y0_candidates": Y0_COVERAGE_CANDIDATES,
        "candidate_clean": candidate_clean,
        "candidate_coverage_fractions": candidate_coverage_fractions,
        "candidate_coverage_groups": candidate_coverage_groups,
        "n_covered_ics": len(covered_y0),
        "n_partial_ics": len(partial_y0),
        "n_uncovered_ics": len(uncovered_y0),
        "coverage_radius": coverage_radius,
        "coverage_fraction_high": COVERAGE_FRACTION_HIGH,
        "coverage_fraction_low": COVERAGE_FRACTION_LOW,
        "coverage_eval_horizon": COVERAGE_EVAL_HORIZON,
        "state_cloud_size": int(state_cloud.shape[0]),
    }


def train_one_interval(
    seed,
    train_interval,
    ratio,
    data,
    chunk_size,
    stage_scale,
    regularization_profile,
):
    config_name = build_config_name(train_interval, ratio)
    params = initial_params(seed)
    batch_loss = make_batch_loss(TRAINING_METHOD, ratio, data["obs_dt"])
    training_objective = make_training_objective(batch_loss)

    def make_train_chunk(optimizer, num_steps):
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
                length=num_steps,
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
        params = {**params, "residual_scale": jnp.array(stage["residual_scale"])}
        opt_state = optimizer.init(params)
        l2_weight = jnp.array(regularization_profile["l2_weight"])
        ortho_weight = jnp.array(regularization_profile["ortho_weight"])
        full_chunks, remainder = divmod(stage["epochs"], chunk_size)
        chunk_plan = [chunk_size] * full_chunks
        if remainder:
            chunk_plan.append(remainder)
        train_chunks = {
            steps: make_train_chunk(optimizer, steps) for steps in set(chunk_plan)
        }
        progress(
            f"[train] {config_name}, seed={seed}, {stage['name']}, "
            f"chunks={len(chunk_plan)}"
        )

        for steps_in_chunk in chunk_plan:
            params, opt_state, losses = train_chunks[steps_in_chunk](
                params,
                opt_state,
                l2_weight,
                ortho_weight,
            )
            global_epoch += steps_in_chunk
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
                    "l2_weight": regularization_profile["l2_weight"],
                    "ortho_weight": regularization_profile["ortho_weight"],
                }
            )

            if best_loss - validation_loss > min_delta:
                best_loss = validation_loss
                best_params = params
                best_epoch = global_epoch
                best_stage = stage["name"]
                stage_counter = 0
            else:
                stage_counter += steps_in_chunk

            if stage["patience"] is not None and stage_counter >= stage["patience"]:
                break

    return {
        "params": best_params,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "training_config": config_name,
        "training_method": TRAINING_METHOD_NAME,
        "train_interval": train_interval,
        "ratio": ratio,
        "h_model": data["obs_dt"] / ratio,
        "seed": seed,
        "regularization_profile": regularization_profile["name"],
        "l2_weight": regularization_profile["l2_weight"],
        "ortho_weight": regularization_profile["ortho_weight"],
        "chunk_size": chunk_size,
        "stage_scale": stage_scale,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "stage_history": stage_history,
    }


def batch_mse_slice(pred, true, start_idx=None, end_idx=None):
    if pred.shape[0] == 0:
        return float("nan")
    if start_idx is None and end_idx is None:
        pred_slice = pred
        true_slice = true
    elif end_idx is None:
        pred_slice = pred[:, start_idx:, :]
        true_slice = true[:, start_idx:, :]
    else:
        pred_slice = pred[:, start_idx:end_idx, :]
        true_slice = true[:, start_idx:end_idx, :]
    return float(jnp.mean((pred_slice - true_slice) ** 2))


def evaluate_interval(trained, data):
    params = trained["params"]
    t_obs = data["t_obs"]
    t_final_grid = data["t_final_grid"]
    T_train = data["T_train"]
    extrap_start = data["extrap_start"]
    obs_dt = data["obs_dt"]

    pred_train = solve_model_diffrax_batch(y0_batch, params, t_obs, T_train)
    pred_val = solve_model_diffrax_batch(y0_validation, params, t_obs, T_train)
    pred_extrap = solve_model_diffrax_batch(
        y0_batch,
        params,
        t_final_grid,
        data["T_final"],
    )
    pred_candidates = solve_model_diffrax_batch(
        data["y0_candidates"],
        params,
        t_final_grid,
        data["T_final"],
    )

    if data["y0_covered"].shape[0] > 0:
        pred_covered = solve_model_diffrax_batch(
            data["y0_covered"],
            params,
            t_final_grid,
            data["T_final"],
        )
    else:
        pred_covered = jnp.zeros((0, t_final_grid.shape[0], 2))

    if data["y0_partial"].shape[0] > 0:
        pred_partial = solve_model_diffrax_batch(
            data["y0_partial"],
            params,
            t_final_grid,
            data["T_final"],
        )
    else:
        pred_partial = jnp.zeros((0, t_final_grid.shape[0], 2))

    if data["y0_uncovered"].shape[0] > 0:
        pred_uncovered = solve_model_diffrax_batch(
            data["y0_uncovered"],
            params,
            t_final_grid,
            data["T_final"],
        )
    else:
        pred_uncovered = jnp.zeros((0, t_final_grid.shape[0], 2))

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

    horizon_metrics = {
        horizon_field_name(horizon): horizon_mse(
            pred_extrap,
            data["extrap"],
            extrap_start,
            horizon,
            obs_dt,
        )
        for horizon in FIXED_EXTRAP_HORIZONS
    }
    candidate_horizon_values = {
        horizon: candidate_horizon_mse(
            pred_candidates,
            data["candidate_clean"],
            extrap_start,
            horizon,
            obs_dt,
        )
        for horizon in FIXED_EXTRAP_HORIZONS
    }
    candidate_horizon_metrics = {
        f"candidate_ic_extrap_mse_h{f'{horizon:g}'.replace('.', 'p')}": float(
            jnp.mean(values)
        )
        for horizon, values in candidate_horizon_values.items()
    }

    row = {
        "experiment_kind": "carrying_train_interval_sweep",
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "training_protocol_version": trained["training_protocol_version"],
        "chunk_size": trained["chunk_size"],
        "stage_scale": trained["stage_scale"],
        "evaluation_status": "ok",
        "evaluation_error_type": "",
        "evaluation_error": "",
        "solver_failed": False,
        "training_config": trained["training_config"],
        "training_method": trained["training_method"],
        "train_interval": trained["train_interval"],
        "ratio": trained["ratio"],
        "seed": trained["seed"],
        "regularization_profile": trained["regularization_profile"],
        "l2_weight": trained["l2_weight"],
        "ortho_weight": trained["ortho_weight"],
        "h_model": trained["h_model"],
        "num_observed": data["num_observed"],
        "obs_dt": data["obs_dt"],
        "T_final": data["T_final"],
        "evaluation_method": "diffrax_tsit5",
        "noise_level": data.get("noise_level", DEFAULT_NOISE_LEVEL),
        "coverage_radius": data["coverage_radius"],
        "coverage_fraction_high": data["coverage_fraction_high"],
        "coverage_fraction_low": data["coverage_fraction_low"],
        "coverage_eval_horizon": data["coverage_eval_horizon"],
        "n_covered_ics": data["n_covered_ics"],
        "n_partial_ics": data["n_partial_ics"],
        "n_uncovered_ics": data["n_uncovered_ics"],
        "best_loss": trained["best_loss"],
        "best_epoch": trained["best_epoch"],
        "best_stage": trained["best_stage"],
        "train_mse": float(jnp.mean((pred_train - data["train_clean"]) ** 2)),
        "validation_mse": float(jnp.mean((pred_val - data["val_clean"]) ** 2)),
        "same_ic_extrapolate_mse": batch_mse_slice(
            pred_extrap,
            data["extrap"],
            extrap_start,
            None,
        ),
        "extrapolate_mse": batch_mse_slice(
            pred_extrap,
            data["extrap"],
            extrap_start,
            None,
        ),
        "covered_ic_rollout_mse": batch_mse_slice(pred_covered, data["covered_clean"]),
        "covered_ic_extrapolate_mse": batch_mse_slice(
            pred_covered,
            data["covered_clean"],
            extrap_start,
            None,
        ),
        "partial_ic_rollout_mse": batch_mse_slice(pred_partial, data["partial_clean"]),
        "partial_ic_extrapolate_mse": batch_mse_slice(
            pred_partial,
            data["partial_clean"],
            extrap_start,
            None,
        ),
        "uncovered_ic_rollout_mse": batch_mse_slice(pred_uncovered, data["uncovered_clean"]),
        "uncovered_ic_extrapolate_mse": batch_mse_slice(
            pred_uncovered,
            data["uncovered_clean"],
            extrap_start,
            None,
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
        **horizon_metrics,
        **candidate_horizon_metrics,
        **vector_field_metrics_with_coverage(
            params,
            data["state_cloud"],
            data["coverage_radius"],
        ),
    }
    curves = {
        "y0_covered": np.asarray(data["y0_covered"]),
        "y0_partial": np.asarray(data["y0_partial"]),
        "y0_uncovered": np.asarray(data["y0_uncovered"]),
        "candidate_y0": np.asarray(data["y0_candidates"]),
        "candidate_coverage_fraction_h10": np.asarray(
            data["candidate_coverage_fractions"]
        ),
        "candidate_coverage_group_h10": np.asarray(
            data["candidate_coverage_groups"]
        ),
        "candidate_extrap_mse_h2p5": np.asarray(candidate_horizon_values[2.5]),
        "candidate_extrap_mse_h5": np.asarray(candidate_horizon_values[5.0]),
        "candidate_extrap_mse_h10": np.asarray(candidate_horizon_values[10.0]),
        "train_mse_by_time": train_mse_by_time,
        "train_rel_error_by_time": train_rel_by_time,
        "train_mse_by_time_state": train_mse_by_time_state,
        "validation_mse_by_time": val_mse_by_time,
        "validation_rel_error_by_time": val_rel_by_time,
        "validation_mse_by_time_state": val_mse_by_time_state,
        "extrapolate_mse_by_time": extrap_mse_by_time,
        "extrapolate_rel_error_by_time": extrap_rel_by_time,
        "extrapolate_mse_by_time_state": extrap_mse_by_time_state,
        "covered_ic_rollout_mse_by_time": (
            np.asarray(jnp.mean((pred_covered - data["covered_clean"]) ** 2, axis=(0, 2)))
            if pred_covered.shape[0] > 0
            else np.array([])
        ),
        "partial_ic_rollout_mse_by_time": (
            np.asarray(jnp.mean((pred_partial - data["partial_clean"]) ** 2, axis=(0, 2)))
            if pred_partial.shape[0] > 0
            else np.array([])
        ),
        "uncovered_ic_rollout_mse_by_time": (
            np.asarray(jnp.mean((pred_uncovered - data["uncovered_clean"]) ** 2, axis=(0, 2)))
            if pred_uncovered.shape[0] > 0
            else np.array([])
        ),
    }
    return row, curves


def evaluation_failure_row(trained, data, error):
    message = " ".join(str(error).split())
    message_lower = message.lower()
    solver_failed = any(
        marker in message_lower
        for marker in ("solver", "maximum number of solver steps", "max_steps")
    )
    return {
        "experiment_kind": "carrying_train_interval_sweep",
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "training_protocol_version": trained["training_protocol_version"],
        "chunk_size": trained["chunk_size"],
        "stage_scale": trained["stage_scale"],
        "evaluation_status": "failed",
        "evaluation_error_type": type(error).__name__,
        "evaluation_error": message[:1000],
        "solver_failed": solver_failed,
        "training_config": trained["training_config"],
        "training_method": trained["training_method"],
        "train_interval": trained["train_interval"],
        "ratio": trained["ratio"],
        "seed": trained["seed"],
        "regularization_profile": trained["regularization_profile"],
        "l2_weight": trained["l2_weight"],
        "ortho_weight": trained["ortho_weight"],
        "h_model": trained["h_model"],
        "num_observed": data["num_observed"],
        "obs_dt": data["obs_dt"],
        "T_final": data["T_final"],
        "evaluation_method": "diffrax_tsit5",
        "noise_level": data.get("noise_level", DEFAULT_NOISE_LEVEL),
        "coverage_radius": data["coverage_radius"],
        "coverage_fraction_high": data["coverage_fraction_high"],
        "coverage_fraction_low": data["coverage_fraction_low"],
        "coverage_eval_horizon": data["coverage_eval_horizon"],
        "n_covered_ics": data["n_covered_ics"],
        "n_partial_ics": data["n_partial_ics"],
        "n_uncovered_ics": data["n_uncovered_ics"],
        "best_loss": trained["best_loss"],
        "best_epoch": trained["best_epoch"],
        "best_stage": trained["best_stage"],
    }


SUMMARY_FIELDS = [
    "experiment_kind",
    "evaluation_schema_version",
    "training_protocol_version",
    "chunk_size",
    "stage_scale",
    "evaluation_status",
    "evaluation_error_type",
    "evaluation_error",
    "solver_failed",
    "training_config",
    "training_method",
    "train_interval",
    "ratio",
    "seed",
    "regularization_profile",
    "l2_weight",
    "ortho_weight",
    "h_model",
    "num_observed",
    "obs_dt",
    "T_final",
    "evaluation_method",
    "noise_level",
    "coverage_radius",
    "coverage_fraction_high",
    "coverage_fraction_low",
    "coverage_eval_horizon",
    "n_covered_ics",
    "n_partial_ics",
    "n_uncovered_ics",
    "best_loss",
    "best_epoch",
    "best_stage",
    "train_mse",
    "validation_mse",
    "same_ic_extrapolate_mse",
    "same_ic_extrap_mse_h2p5",
    "same_ic_extrap_mse_h5",
    "same_ic_extrap_mse_h10",
    "candidate_ic_extrap_mse_h2p5",
    "candidate_ic_extrap_mse_h5",
    "candidate_ic_extrap_mse_h10",
    "extrapolate_mse",
    "covered_ic_rollout_mse",
    "covered_ic_extrapolate_mse",
    "partial_ic_rollout_mse",
    "partial_ic_extrapolate_mse",
    "uncovered_ic_rollout_mse",
    "uncovered_ic_extrapolate_mse",
    "final_mse",
    "final_l2_error",
    "instability_rate",
    "residual_mse",
    "residual_rel_error",
    "model_vf_mse",
    "model_vf_rel_error",
    "model_vf_mse_seen",
    "model_vf_rel_error_seen",
    "model_vf_mse_unseen",
    "model_vf_rel_error_unseen",
    "model_vf_mse_training_states",
    "model_vf_rel_error_training_states",
    "vf_grid_n_seen",
    "vf_grid_n_unseen",
    "learned_f_physics_a",
    "learned_f_physics_b",
    "learned_f_physics_r",
    "learned_f_physics_z",
]


def write_summary_csv(rows, path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(output_dir):
    summary_path = output_dir / "carrying_train_interval_sweep_summary.csv"
    details_path = output_dir / "carrying_train_interval_sweep_details.pkl"
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

    def compatible_key(key):
        return (
            isinstance(key, tuple)
            and len(key) == 8
            and int(key[-1]) == TRAINING_PROTOCOL_VERSION
        )

    details = {key: value for key, value in details.items() if compatible_key(key)}
    trained_models = {
        key: value for key, value in trained_models.items() if compatible_key(key)
    }

    # Results produced by the earlier evaluation protocol are intentionally not
    # resumed: their variable extrapolation windows and candidate groups are not
    # comparable to schema v2.
    rows = [
        row
        for row in rows
        if parse_int(row.get("evaluation_schema_version"))
        == EVALUATION_SCHEMA_VERSION
        and parse_int(row.get("training_protocol_version"))
        == TRAINING_PROTOCOL_VERSION
        and parse_int(row.get("chunk_size")) is not None
        and np.isfinite(parse_float(row.get("stage_scale")))
    ]

    completed = {
        row_run_key(row)
        for row in rows
        if row.get("train_interval") not in (None, "")
        and row.get("seed") not in (None, "")
        and row.get("evaluation_status") == "ok"
    }
    return rows, details, trained_models, completed


def save_results(
    output_dir,
    rows,
    details,
    trained_models,
    train_intervals,
    ratio,
    noise_level,
    coverage_radius,
    chunk_size,
    stage_scale,
):
    write_summary_csv(rows, output_dir / "carrying_train_interval_sweep_summary.csv")
    with (output_dir / "carrying_train_interval_sweep_details.pkl").open("wb") as f:
        pickle.dump(
            {
                "trained": trained_models,
                "rows": rows,
                "details": details,
                "train_intervals": train_intervals,
                "ratio": ratio,
                "noise_level": noise_level,
                "coverage_radius": coverage_radius,
                "coverage_fraction_high": COVERAGE_FRACTION_HIGH,
                "coverage_fraction_low": COVERAGE_FRACTION_LOW,
                "coverage_eval_horizon": COVERAGE_EVAL_HORIZON,
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                "chunk_size": chunk_size,
                "stage_scale": stage_scale,
                "regularization_profile": REGULARIZATION_PROFILE,
                "obs_dt": OBS_DT,
                "T_final": T_FINAL,
            },
            f,
        )


def run_train_interval_sweep(
    output_dir,
    seeds,
    train_intervals,
    ratio,
    noise_level,
    coverage_radius,
    chunk_size,
    stage_scale,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_sweep_args(
        train_intervals,
        ratio,
        noise_level,
        coverage_radius,
        chunk_size,
        stage_scale,
    )
    manifest_parameters = {
        "seeds": list(seeds),
        "train_intervals": list(train_intervals),
        "ratio": ratio,
        "noise_level": noise_level,
        "coverage_radius": coverage_radius,
        "coverage_eval_horizon": COVERAGE_EVAL_HORIZON,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "chunk_size": chunk_size,
        "stage_scale": stage_scale,
        "regularization_profile": REGULARIZATION_PROFILE,
        "obs_dt": OBS_DT,
        "t_final": T_FINAL,
    }
    write_run_manifest(
        output_dir,
        "carrying_train_interval_sweep",
        manifest_parameters,
        filename="run_manifest.json",
    )
    write_run_manifest(
        output_dir,
        "carrying_train_interval_sweep_evaluation",
        manifest_parameters,
        filename="evaluation_manifest.json",
    )

    progress(f"[setup] writing train-interval sweep to {output_dir}")
    progress(
        f"[setup] train_intervals={train_intervals}, ratio={ratio}, "
        f"seeds={len(seeds)}, noise={noise_level}, coverage_radius={coverage_radius}"
    )

    rows, details, trained_models, completed = load_existing_results(output_dir)
    training_manifest_path = output_dir / "training_manifest.json"
    if not training_manifest_path.exists() and not trained_models:
        write_run_manifest(
            output_dir,
            "carrying_train_interval_sweep_training",
            manifest_parameters,
            filename="training_manifest.json",
        )
    elif not training_manifest_path.exists():
        progress(
            "[warning] compatible trained models exist without a training manifest; "
            "the script will not fabricate their training provenance"
        )
    if completed:
        progress(f"[resume] found {len(completed)} completed runs; skipping them")

    total = len(train_intervals) * len(seeds)
    run_idx = 0
    for train_interval in train_intervals:
        progress(f"[setup] building reference data for T_train={train_interval:g}")
        data = make_reference_data_for_interval(
            train_interval,
            noise_level=noise_level,
            coverage_radius=coverage_radius,
        )
        data["noise_level"] = noise_level
        progress(
            f"[setup] T_train={train_interval:g}, num_observed={data['num_observed']}, "
            f"covered_ics={data['n_covered_ics']}, partial_ics={data['n_partial_ics']}, "
            f"uncovered_ics={data['n_uncovered_ics']}"
        )

        for seed in seeds:
            run_idx += 1
            key = run_key(
                train_interval,
                ratio,
                noise_level,
                coverage_radius,
                seed,
                chunk_size,
                stage_scale,
            )
            if key in completed:
                progress(
                    f"[run {run_idx}/{total}] skip completed "
                    f"T_train={train_interval:g}, seed={seed}"
                )
                continue

            config_name = build_config_name(train_interval, ratio)
            if key in trained_models:
                progress(
                    f"[run {run_idx}/{total}] reuse trained model and re-evaluate "
                    f"{config_name}, seed={seed}"
                )
                trained = trained_models[key]
            else:
                progress(
                    f"[run {run_idx}/{total}] train {config_name}, seed={seed}"
                )
                trained = train_one_interval(
                    seed=seed,
                    train_interval=train_interval,
                    ratio=ratio,
                    data=data,
                    chunk_size=chunk_size,
                    stage_scale=stage_scale,
                    regularization_profile=REGULARIZATION_PROFILE,
                )
                trained_models[key] = trained
                # Persist the expensive training result before evaluation.  A
                # divergent rollout can then be retried without retraining.
                save_results(
                    output_dir,
                    rows,
                    details,
                    trained_models,
                    train_intervals,
                    ratio,
                    noise_level,
                    coverage_radius,
                    chunk_size,
                    stage_scale,
                )

            progress(
                f"[run {run_idx}/{total}] eval {config_name}, seed={seed}, "
                f"best_loss={trained['best_loss']:.6e}"
            )
            try:
                row, curves = evaluate_interval(trained, data)
            except Exception as error:
                row = evaluation_failure_row(trained, data, error)
                curves = None
                progress(
                    f"[run {run_idx}/{total}] evaluation failed "
                    f"{config_name}, seed={seed}: "
                    f"{row['evaluation_error_type']}: {row['evaluation_error']}"
                )

            # Replace an earlier failed evaluation for the same training run.
            rows = [existing for existing in rows if row_run_key(existing) != key]
            rows.append(row)
            if curves is None:
                details.pop(key, None)
            else:
                details[key] = curves
            if row["evaluation_status"] != "ok":
                save_results(
                    output_dir,
                    rows,
                    details,
                    trained_models,
                    train_intervals,
                    ratio,
                    noise_level,
                    coverage_radius,
                    chunk_size,
                    stage_scale,
                )
                continue
            progress(
                f"[run {run_idx}/{total}] done {config_name}, seed={seed}, "
                f"same_ic_extrap={row['same_ic_extrapolate_mse']:.6e}, "
                f"covered_extrap={row['covered_ic_extrapolate_mse']:.6e}, "
                f"partial_extrap={row['partial_ic_extrapolate_mse']:.6e}, "
                f"uncovered_extrap={row['uncovered_ic_extrapolate_mse']:.6e}, "
                f"vf_seen={row['model_vf_rel_error_seen']:.6e}, "
                f"vf_unseen={row['model_vf_rel_error_unseen']:.6e}"
            )
            save_results(
                output_dir,
                rows,
                details,
                trained_models,
                train_intervals,
                ratio,
                noise_level,
                coverage_radius,
                chunk_size,
                stage_scale,
            )

    save_results(
        output_dir,
        rows,
        details,
        trained_models,
        train_intervals,
        ratio,
        noise_level,
        coverage_radius,
        chunk_size,
        stage_scale,
    )
    progress("[done] carrying train-interval sweep complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep training time horizon T_train while fixing RK4 ratio, no "
            "regularization, and zero noise. Evaluates same-IC extrapolation, "
            "covered new-IC, and uncovered new-IC generalization."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="carrying_train_interval_sweep_results",
    )
    parser.add_argument(
        "--train-intervals",
        nargs="+",
        type=float,
        default=DEFAULT_TRAIN_INTERVALS,
    )
    parser.add_argument("--ratio", type=int, default=DEFAULT_RATIO)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL)
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS,
        help="Distance threshold for classifying new ICs as covered by training trajectories.",
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
    run_train_interval_sweep(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_intervals=args.train_intervals,
        ratio=args.ratio,
        noise_level=args.noise_level,
        coverage_radius=args.coverage_radius,
        chunk_size=args.chunk_size,
        stage_scale=args.stage_scale,
    )
