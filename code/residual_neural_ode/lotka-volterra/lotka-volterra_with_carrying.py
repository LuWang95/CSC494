import jax.numpy as jnp
import optax
import diffrax
import pickle
from jax import jit, lax, value_and_grad, vmap
from jax import random
import jax.nn as jnn


from solver import*
from functools import partial
from code.error_propagation.visualize_lotka_volterra import visualize_results



# -----------------------------

# Lotka-Volterra model (x = prey, y = predator)
# dx/dt = a x - b x y - c x ^ 2
# dy/dt = -r y + z x y
# state y = [prey, predator]
# -----------------------------

a = 1
b = 0.05
r = 1.5
z = 0.03
c = 0.001

noise_level = 0.01

y0_batch = jnp.array([
    [15.0, 25.0],
    [10.0, 20.0],
    [20.0, 30.0],
    [8.0, 18.0],
    [22.0, 26.0],
    [16.0, 15.0],
    # high prey
    [35.0, 20.0],
    [40.0, 35.0],
    [45.0, 15.0],
    # low prey
    [3.0, 20.0],
    [5.0, 10.0],
    # high predator
    [15.0, 45.0],
    [10.0, 50.0],
    # low predator
    [20.0, 5.0],
    [30.0, 8.0],
    # both high
    [40.0, 40.0],
    # prey high predator low
    [45.0, 5.0],
    # prey low predator high
    [5.0, 45.0],

])

y0_validation = jnp.array([
    [12.0, 35.0],
    [25.0, 12.0],
    [32.0, 28.0],
    [7.0, 35.0],
])

T = 5
num_observed = 101
t_obs = jnp.linspace(0.0, T, num_observed)
T_extrapolate = 4 * T
num_extrapolate_observed = 4 * (num_observed - 1) + 1
t_extrapolate = jnp.linspace(0.0, T_extrapolate, num_extrapolate_observed)
ratio = 10
h_model = (t_obs[1] - t_obs[0]) / ratio

def lotka_volterra(t, y, args):
    prey, predator = y[0], y[1]
    dxdt = a * prey - b * prey * predator - c * prey * prey
    dydt = -r * predator + z * prey * predator
    return jnp.array([dxdt, dydt])


def solve_reference(y0):
    term = diffrax.ODETerm(lotka_volterra)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_obs)
    stepsize_controller = diffrax.PIDController(
        rtol=1e-9,
        atol=1e-9

    )
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000
    )
    return sol.ys

def solve_reference_extrapolate(y0):
    term = diffrax.ODETerm(lotka_volterra)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_extrapolate)
    stepsize_controller = diffrax.PIDController(
        rtol=1e-9,
        atol=1e-9
    )
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T_extrapolate,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000
    )
    return sol.ys

observed_batch_clean = vmap(solve_reference)(y0_batch)
observed_validation_clean = vmap(solve_reference)(y0_validation)
key = random.key(42)
train_noise = (
    noise_level
    * observed_batch_clean
    * random.normal(key, observed_batch_clean.shape)
)
observed_batch = observed_batch_clean + train_noise
observed_validation = observed_validation_clean
print(observed_batch.shape)
print(observed_validation.shape)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes)-1)
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


state_dim = 2
layer_sizes = [state_dim, 64, 64, state_dim]
step_size = 3e-3
num_epochs = 30000
nn_params = init_network_params(layer_sizes, random.key(0))
# incomplete physics: linear growth/decay only (missing xy interaction)
f_physics_params = jnp.array([0.7, 0.03, 1.0, 0.025])
params = {
    "nn_params": nn_params,
    "f_physics": f_physics_params,
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

optimizer = make_optimizer(step_size, step_size)
opt_state = optimizer.init(params)


state_scale = jnp.array([50.0, 50.0])
def nn(y, nn_parameters):
    activations = y / state_scale
    for w, b in nn_parameters[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jnn.swish(outputs)
    final_w, final_b = nn_parameters[-1]
    return jnp.dot(final_w, activations) + final_b

## fphysics only learns linear growth/decay
def f_physics(y, f_physics_params):
    prey, predator = y[0], y[1]
    return jnp.array([
        f_physics_params[0] * prey - f_physics_params[1] * predator * prey,
        -f_physics_params[2] * predator + f_physics_params[3] * predator * prey,
    ])

def model_rhs(y, params):
    return (
        f_physics(y, params["f_physics"])
        + params["residual_scale"] * nn(y, params["nn_params"])
    )

@partial(jit, static_argnames=("step_ratio",))
def loss(parameters, y0, true_trajectory, h, step_ratio):
    num_steps = (true_trajectory.shape[0] - 1) * step_ratio
    pred_full = roll_out(y0,h,model_rhs,parameters,num_steps,rk4)
    pred_useful = pred_full[::step_ratio]
    return jnp.mean((true_trajectory - pred_useful)**2)


def l2_regularization(parameters, y0):
    residuals = vmap(nn,in_axes=(0,None))(y0, parameters["nn_params"])
    return jnp.mean(residuals**2)

def squared_correlation(u, v):
    u = u - jnp.mean(u)
    v = v - jnp.mean(v)
    return (jnp.sum(u * v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-8)) ** 2

def orthogonality_regularization(parameters, states):
    residuals = vmap(nn, in_axes=(0, None))(states, parameters["nn_params"])
    x = states[:, 0]
    y = states[:, 1]
    xy = x * y
    r1 = residuals[:, 0]
    r2 = residuals[:, 1]
    reg = (squared_correlation(r1, x) + squared_correlation(r1, xy) + squared_correlation(r2, y) + squared_correlation(r2, xy))
    return reg

@partial(jit, static_argnames=("step_ratio",))
def batch_loss(parameters, y0_batch, true_trajectories, h, step_ratio):
    losses = vmap(loss, in_axes=(None,0,0,None,None))(parameters, y0_batch, true_trajectories, h, step_ratio )
    data_loss = jnp.mean(losses)
    return data_loss


@partial(jit, static_argnames=("step_ratio",))
def training_objective(
    parameters,
    y0,
    true_trajectories,
    h,
    step_ratio,
    l2_weight,
    ortho_weight,
):
    loss = batch_loss(parameters, y0, true_trajectories, h, step_ratio)
    l2_term = l2_regularization(parameters, true_trajectories.reshape(-1, 2))
    ortho_term = orthogonality_regularization(parameters, true_trajectories.reshape(-1, 2))
    return loss + l2_weight * l2_term + ortho_weight * ortho_term

def train_step(stage_optimizer):
    def step(carry, _):
        parameters, opt_state, l2_weight, ortho_weight = carry
        loss_val, grads = value_and_grad(training_objective)(
            parameters,
            y0_batch,
            observed_batch,
            h_model,
            ratio,
            l2_weight,
            ortho_weight,
        )
        updates, opt_state = stage_optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return (parameters, opt_state, l2_weight, ortho_weight), loss_val

    return step

chunk_size = 100

def make_train_chunk(stage_optimizer):
    @jit
    def train_chunk(parameters, opt_state, l2_weight, ortho_weight):
        (parameters, opt_state, _, _), losses = lax.scan(
            train_step(stage_optimizer),
            (parameters, opt_state, l2_weight, ortho_weight),
            None,
            length=chunk_size,
        )
        return parameters, opt_state, losses

    return train_chunk

best_loss = float("inf")
best_params = params
min_delta = 1e-7
best_stage = ""
best_epoch = 0
best_l2_weight = 0.0
best_ortho_weight = 0.0
best_nn_lr = step_size
best_physics_lr = step_size
best_residual_scale = 1.0

training_stages = [
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
        "l2_weight": 0,
        "ortho_weight": 0,
        "patience": 1000,
    },
    {
        "name": "stage 3 | joint training",
        "epochs": 8000,
        "nn_lr": 1e-3,
        "physics_lr": 2e-4,
        "residual_scale": 1.0,
        "l2_weight": 0,
        "ortho_weight": 0,
        "patience": 1200,
    },
    {
        "name": "stage 4 | physics fine tune",
        "epochs": 1500,
        "nn_lr": 0.0,
        "physics_lr": 0,
        "residual_scale": 1.0,
        "l2_weight": 0.0,
        "ortho_weight": 0.0,
        "patience": 500,
    },
]

global_epoch = 0
stage_history = []

for stage in training_stages:
    stage_counter = 0
    stage_chunks = stage["epochs"] // chunk_size
    l2_weight = jnp.array(stage["l2_weight"])
    ortho_weight = jnp.array(stage["ortho_weight"])
    params = {
        **params,
        "residual_scale": jnp.array(stage["residual_scale"]),
    }
    optimizer = make_optimizer(stage["nn_lr"], stage["physics_lr"])
    opt_state = optimizer.init(params)
    train_chunk = make_train_chunk(optimizer)
    print(
        f"\n{stage['name']}: "
        f"nn_lr={stage['nn_lr']}, physics_lr={stage['physics_lr']}, "
        f"residual_scale={stage['residual_scale']}, "
        f"l2={stage['l2_weight']}, ortho={stage['ortho_weight']}"
    )

    for _ in range(stage_chunks):
        params, opt_state, losses = train_chunk(
            params,
            opt_state,
            l2_weight,
            ortho_weight,
        )
        global_epoch += chunk_size
        train_objective = losses[-1]
        train_data_loss = batch_loss(
            params,
            y0_batch,
            observed_batch,
            h_model,
            ratio,
        )
        validation_data_loss = batch_loss(
            params,
            y0_validation,
            observed_validation,
            h_model,
            ratio,
        )
        print(
            f"epoch {global_epoch}, "
            f"objective = {train_objective}, "
            f"train data loss = {train_data_loss}, "
            f"validation data loss = {validation_data_loss}"
        )

        stage_history.append(
            {
                "stage": stage["name"],
                "epoch": global_epoch,
                "train_objective": float(train_objective),
                "train_data_loss": float(train_data_loss),
                "validation_data_loss": float(validation_data_loss),
                "nn_lr": stage["nn_lr"],
                "physics_lr": stage["physics_lr"],
                "residual_scale": stage["residual_scale"],
                "l2_weight": stage["l2_weight"],
                "ortho_weight": stage["ortho_weight"],
            }
        )

        if best_loss - float(validation_data_loss) > min_delta:
            best_loss = float(validation_data_loss)
            best_params = params
            best_stage = stage["name"]
            best_epoch = global_epoch
            best_l2_weight = stage["l2_weight"]
            best_ortho_weight = stage["ortho_weight"]
            best_nn_lr = stage["nn_lr"]
            best_physics_lr = stage["physics_lr"]
            best_residual_scale = stage["residual_scale"]
            stage_counter = 0
        else:
            stage_counter += chunk_size

        if stage["patience"] is not None and stage_counter >= stage["patience"]:
            print(
                f"Early stopping {stage['name']} at epoch {global_epoch}, "
                f"best validation loss = {best_loss}"
            )
            break

params = best_params

# -----------------------------
# Check prediction after training
# -----------------------------
index = 0
validation_index = 0
num_steps = int((observed_batch.shape[1] - 1) * ratio)

pred_full = roll_out(y0_batch[0], h_model, model_rhs, params, num_steps, rk4)
pred_obs = pred_full[::ratio]
pred_batch_full = vmap(
    lambda y0: roll_out(y0, h_model, model_rhs, params, num_steps, rk4)
)(y0_batch)
pred_batch = pred_batch_full[:, ::ratio, :]
pred_validation_full = roll_out(
    y0_validation[validation_index], h_model, model_rhs, params, num_steps, rk4
)
pred_validation_obs = pred_validation_full[::ratio]
pred_validation_batch_full = vmap(
    lambda y0: roll_out(y0, h_model, model_rhs, params, num_steps, rk4)
)(y0_validation)
pred_validation_batch = pred_validation_batch_full[:, ::ratio, :]

num_extrapolate_steps = int((num_extrapolate_observed - 1) * ratio)
true_extrapolate_batch = vmap(solve_reference_extrapolate)(y0_batch)
pred_extrapolate_full = roll_out(
    y0_batch[index], h_model, model_rhs, params, num_extrapolate_steps, rk4
)
pred_extrapolate_obs = pred_extrapolate_full[::ratio]
pred_extrapolate_batch_full = vmap(
    lambda y0: roll_out(y0, h_model, model_rhs, params, num_extrapolate_steps, rk4)
)(y0_batch)
pred_extrapolate_batch = pred_extrapolate_batch_full[:, ::ratio, :]

## phase portrait: true vs learned vector field
prey_min, prey_max = 5.0, 25.0
pred_min, pred_max = 10.0, 40.0
nx, ny = 20, 20
prey_vals = jnp.linspace(prey_min, prey_max, nx)
pred_vals = jnp.linspace(pred_min, pred_max, ny)
Prey, Pred = jnp.meshgrid(prey_vals, pred_vals)
states = jnp.stack([Prey.reshape(-1), Pred.reshape(-1)], axis=1)

def true_lv_rhs(state):
    return lotka_volterra(0.0, state, None)

true_vf = vmap(true_lv_rhs)(states)
physics_vf = vmap(lambda s: f_physics(s, params["f_physics"]))(states)
nn_vf = vmap(lambda s: params["residual_scale"] * nn(s, params["nn_params"]))(states)
model_vf = vmap(lambda s: model_rhs(s, params))(states)

test_states = jnp.array([
    [15.0, 25.0],
    [10.0, 20.0],
    [20.0, 30.0],
    [16.0, 15.0],
])
sample_true_vf = vmap(true_lv_rhs)(test_states)
sample_physics_vf = vmap(lambda s: f_physics(s, params["f_physics"]))(test_states)
sample_nn_residual = vmap(
    lambda s: params["residual_scale"] * nn(s, params["nn_params"])
)(test_states)
sample_model_vf = vmap(lambda s: model_rhs(s, params))(test_states)
sample_true_residual = sample_true_vf - sample_physics_vf

# -----------------------------
# Metrics for saved results
# -----------------------------
eps = 1e-8
true_residual = true_vf - physics_vf
residual_mse = jnp.mean((nn_vf - true_residual) ** 2)
model_vf_mse = jnp.mean((model_vf - true_vf) ** 2)
residual_rel_error = (
    jnp.linalg.norm(nn_vf - true_residual) / (jnp.linalg.norm(true_residual) + eps)
)

model_vf_rel_error = (
    jnp.linalg.norm(model_vf - true_vf) / (jnp.linalg.norm(true_vf) + eps)

)
train_clean_loss = batch_loss(params, y0_batch, observed_batch_clean, h_model, ratio)
train_noisy_loss = batch_loss(params, y0_batch, observed_batch, h_model, ratio)
train_l2_regularization = l2_regularization(params, observed_batch.reshape(-1, 2))
train_orthogonality_regularization = orthogonality_regularization(
    params,
    observed_batch.reshape(-1, 2),
)
train_regularized_objective = (
    train_noisy_loss
    + best_l2_weight * train_l2_regularization
    + best_ortho_weight * train_orthogonality_regularization
)
train_sample_clean_mse = jnp.mean((pred_obs - observed_batch_clean[index]) ** 2)
train_sample_clean_rel_error = (
    jnp.linalg.norm(pred_obs - observed_batch_clean[index])
    / (jnp.linalg.norm(observed_batch_clean[index]) + eps)
)
train_sample_noisy_mse = jnp.mean((pred_obs - observed_batch[index]) ** 2)
train_sample_noisy_rel_error = (
    jnp.linalg.norm(pred_obs - observed_batch[index])
    / (jnp.linalg.norm(observed_batch[index]) + eps)
)
train_clean_batch_mse = jnp.mean((pred_batch - observed_batch_clean) ** 2)
train_clean_batch_rel_error = (
    jnp.linalg.norm(pred_batch - observed_batch_clean)
    / (jnp.linalg.norm(observed_batch_clean) + eps)
)
train_noisy_batch_mse = jnp.mean((pred_batch - observed_batch) ** 2)
train_noisy_batch_rel_error = (
    jnp.linalg.norm(pred_batch - observed_batch)
    / (jnp.linalg.norm(observed_batch) + eps)
)
validation_loss = batch_loss(params, y0_validation, observed_validation, h_model, ratio)
validation_sample_mse = jnp.mean(
    (pred_validation_obs - observed_validation[validation_index]) ** 2
)
validation_sample_rel_error = (
    jnp.linalg.norm(pred_validation_obs - observed_validation[validation_index])
    / (jnp.linalg.norm(observed_validation[validation_index]) + eps)
)
validation_batch_mse = jnp.mean((pred_validation_batch - observed_validation) ** 2)
validation_batch_rel_error = (
    jnp.linalg.norm(pred_validation_batch - observed_validation)
    / (jnp.linalg.norm(observed_validation) + eps)
)
extrapolate_metric_start = num_observed
extrapolate_sample_mse = jnp.mean(
    (
        pred_extrapolate_obs[extrapolate_metric_start:]
        - true_extrapolate_batch[index, extrapolate_metric_start:]
    ) ** 2
)
extrapolate_sample_rel_error = (
    jnp.linalg.norm(
        pred_extrapolate_obs[extrapolate_metric_start:]
        - true_extrapolate_batch[index, extrapolate_metric_start:]
    )
    / (
        jnp.linalg.norm(true_extrapolate_batch[index, extrapolate_metric_start:])
        + eps
    )
)
extrapolate_batch_mse = jnp.mean(
    (
        pred_extrapolate_batch[:, extrapolate_metric_start:, :]
        - true_extrapolate_batch[:, extrapolate_metric_start:, :]
    ) ** 2
)
extrapolate_batch_rel_error = (
    jnp.linalg.norm(
        pred_extrapolate_batch[:, extrapolate_metric_start:, :]
        - true_extrapolate_batch[:, extrapolate_metric_start:, :]
    )
    / (
        jnp.linalg.norm(true_extrapolate_batch[:, extrapolate_metric_start:, :])
        + eps
    )
)

results = {
    "experiment_type": "Lotka-Volterra | wrong physics prior | trainable physics params",
    "best_loss": float(best_loss),
    "ratio": ratio,
    "h_model": float(h_model),
    "t_obs": t_obs,
    "t_extrapolate": t_extrapolate,
    "extrapolation_start": float(T),
    "step_size": step_size,
    "training_stages": training_stages,
    "stage_history": stage_history,
    "best_stage": best_stage,
    "best_epoch": best_epoch,
    "best_nn_lr": best_nn_lr,
    "best_physics_lr": best_physics_lr,
    "best_residual_scale": best_residual_scale,
    "best_l2_weight": best_l2_weight,
    "best_ortho_weight": best_ortho_weight,
    "pred_traj": pred_obs,
    "true_traj": observed_batch_clean[index],
    "noisy_train_traj": observed_batch[index],
    "pred_train_batch": pred_batch,
    "true_train_batch": observed_batch_clean,
    "noisy_train_batch": observed_batch,
    "pred_extrapolate_traj": pred_extrapolate_obs,
    "true_extrapolate_traj": true_extrapolate_batch[index],
    "pred_extrapolate_batch": pred_extrapolate_batch,
    "true_extrapolate_batch": true_extrapolate_batch,
    "pred_validation_traj": pred_validation_obs,
    "true_validation_traj": observed_validation[validation_index],
    "pred_validation_batch": pred_validation_batch,
    "true_validation_batch": observed_validation,
    "params": params,
    "learned_f_physics": params["f_physics"],
    "true_vf": true_vf,
    "physics_vf": physics_vf,
    "model_vf": model_vf,
    "true_residual": true_residual,
    "learned_residual": nn_vf,
    "train_clean_loss": float(train_clean_loss),
    "train_noisy_loss": float(train_noisy_loss),
    "train_l2_regularization": float(train_l2_regularization),
    "train_orthogonality_regularization": float(train_orthogonality_regularization),
    "train_regularized_objective": float(train_regularized_objective),
    "sample_traj_mse": float(train_sample_clean_mse),
    "sample_traj_rel_error": float(train_sample_clean_rel_error),
    "train_sample_clean_mse": float(train_sample_clean_mse),
    "train_sample_clean_rel_error": float(train_sample_clean_rel_error),
    "train_sample_noisy_mse": float(train_sample_noisy_mse),
    "train_sample_noisy_rel_error": float(train_sample_noisy_rel_error),
    "train_batch_mse": float(train_clean_batch_mse),
    "train_batch_rel_error": float(train_clean_batch_rel_error),
    "train_clean_batch_mse": float(train_clean_batch_mse),
    "train_clean_batch_rel_error": float(train_clean_batch_rel_error),
    "train_noisy_batch_mse": float(train_noisy_batch_mse),
    "train_noisy_batch_rel_error": float(train_noisy_batch_rel_error),
    "validation_loss": float(validation_loss),
    "validation_sample_mse": float(validation_sample_mse),
    "validation_sample_rel_error": float(validation_sample_rel_error),
    "validation_batch_mse": float(validation_batch_mse),
    "validation_batch_rel_error": float(validation_batch_rel_error),
    "extrapolate_sample_mse": float(extrapolate_sample_mse),
    "extrapolate_sample_rel_error": float(extrapolate_sample_rel_error),
    "extrapolate_batch_mse": float(extrapolate_batch_mse),
    "extrapolate_batch_rel_error": float(extrapolate_batch_rel_error),
    "residual_mse": float(residual_mse),
    "residual_rel_error": float(residual_rel_error),
    "model_vf_mse": float(model_vf_mse),
    "model_vf_rel_error": float(model_vf_rel_error),
    "test_states": test_states,
    "sample_nn_residual": sample_nn_residual,
    "sample_true_residual": sample_true_residual,
    "sample_model_vf": sample_model_vf,
    "sample_true_vf": sample_true_vf,
    "states": states,
}

with open("with_carrying_fixed.pkl", "wb") as f:
    pickle.dump(results, f)

visualize_results(results)