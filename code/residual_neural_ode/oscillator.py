import jax.numpy as jnp
import optax
import diffrax
import pickle
from jax import jit, lax, value_and_grad, vmap
from jax import random
import jax.nn as jnn
from solver import*
from functools import partial
from visualize_oscillator import visualize_results



# -----------------------------

# Duffing oscillator model (x = displacement, v = velocity)
# dx/dt = v
# dv/dt = -rv - ax - bx^3
# state y = [x, v]
# -----------------------------

r = 0.2
a = 1
b = 0.1

y0_batch = jnp.array([
    [-2.0, -1.0],
    [-2.0,  1.0],
    [-1.0, -2.0],
    [-1.0,  2.0],
    [ 1.0, -2.0],
    [ 1.0,  2.0],
    [ 2.0, -1.0],
    [ 2.0,  1.0],
])

y0_validation = jnp.array([[-1.0, -1.0], [1.0, 1.0],[-1.5,-1.5],[1.5,1.5],[0.5,1.0],[-0.5,-1.0]])


T = 5
num_observed = 101
t_obs = jnp.linspace(0.0, T, num_observed)
ratio = 10
h_model = (t_obs[1] - t_obs[0]) / ratio

def duffing(t, y, args):
    x,v = y[0], y[1]
    dx_dt = v
    dv_dt = -r * v -a * x - b*x**3

    return jnp.array([dx_dt, dv_dt])


def solve_reference(y0):
    term = diffrax.ODETerm(duffing)
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
observed_batch = vmap(solve_reference)(y0_batch)
observed_validation = vmap(solve_reference)(y0_validation)
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
f_physics_params = jnp.array([0.1,0.5])
params = {"nn_params": nn_params, "f_physics": f_physics_params}
optimizer = optax.adam(step_size)
opt_state = optimizer.init(params)


state_scale = jnp.array([3.0, 3.0])
def nn(y, nn_parameters):
    activations = y / state_scale
    for w, b in nn_parameters[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jnn.swish(outputs)
    final_w, final_b = nn_parameters[-1]
    return jnp.dot(final_w, activations) + final_b

## fphysics only learns linear growth/decay
def f_physics(y, f_physics_params):
    x, v = y[0], y[1]
    return jnp.array([v, -f_physics_params[0] * x - f_physics_params[1] * v,])

def model_rhs(y, params):
    return f_physics(y, params["f_physics"]) + nn(y, params["nn_params"])

@partial(jit, static_argnames=("step_ratio",))
def loss(parameters, y0, true_trajectory, h, step_ratio):
    num_steps = (true_trajectory.shape[0] - 1) * step_ratio
    pred_full = roll_out(y0,h,model_rhs,parameters,num_steps,rk4)
    pred_useful = pred_full[::step_ratio]
    return jnp.mean((true_trajectory - pred_useful)**2)


@partial(jit, static_argnames=("step_ratio",))
def batch_loss(parameters, y0_batch, true_trajectories, h, step_ratio):
    losses = vmap(loss, in_axes=(None,0,0,None,None))(parameters, y0_batch, true_trajectories, h, step_ratio )
    data_loss = jnp.mean(losses)
    return data_loss

def train_step(carry, _):
    parameters, opt_state = carry
    loss_val, grads = value_and_grad(batch_loss)(
        parameters,
        y0_batch,
        observed_batch,
        h_model,
        ratio,
    )
    updates, opt_state = optimizer.update(grads, opt_state, parameters)
    parameters = optax.apply_updates(parameters, updates)
    return (parameters, opt_state), loss_val

chunk_size = 100

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
best_params = params
patience = 500
min_delta = 1e-7
counter = 0

num_chunks = num_epochs // chunk_size

for chunk in range(num_chunks):
    params, opt_state, losses = train_chunk(params, opt_state)
    epoch = (chunk + 1) * chunk_size
    l = losses[-1]
    print(f"epoch {epoch}, loss = {l}")
    if best_loss - float(l) > min_delta:
        best_loss = float(l)
        best_params = params
        counter = 0
    else:
        counter += chunk_size
    if counter >= patience:
        print(
            f"Early stopping at epoch {epoch}, "
            f"best loss = {best_loss}"
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

## phase portrait: true vs learned vector field
x_min, x_max = -3.0, 3.0
v_min, v_max = -3.0, 3.0
nx, ny = 20, 20
x_vals = jnp.linspace(x_min, x_max, nx)
v_vals = jnp.linspace(v_min, v_max, ny)
X, V = jnp.meshgrid(x_vals, v_vals)
states = jnp.stack([X.reshape(-1), V.reshape(-1)], axis=1)

def true_oscillator_rhs(state):
    return duffing(0.0, state, None)

true_vf = vmap(true_oscillator_rhs)(states)
physics_vf = vmap(lambda s: f_physics(s, params["f_physics"]))(states)
nn_vf = vmap(lambda s: nn(s, params["nn_params"]))(states)
model_vf = vmap(lambda s: model_rhs(s, params))(states)

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
sample_traj_mse = jnp.mean((pred_obs - observed_batch[index]) ** 2)
sample_traj_rel_error = (
    jnp.linalg.norm(pred_obs - observed_batch[index])
    / (jnp.linalg.norm(observed_batch[index]) + eps)
)
train_batch_mse = jnp.mean((pred_batch - observed_batch) ** 2)
train_batch_rel_error = (
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

results = {
    "experiment_type": "Duffing oscillator | wrong physics prior | trainable physics params",
    "best_loss": float(best_loss),
    "ratio": ratio,
    "h_model": float(h_model),
    "t_obs": t_obs,
    "step_size": step_size,
    "pred_traj": pred_obs,
    "true_traj": observed_batch[index],
    "pred_train_batch": pred_batch,
    "true_train_batch": observed_batch,
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
    "sample_traj_mse": float(sample_traj_mse),
    "sample_traj_rel_error": float(sample_traj_rel_error),
    "train_batch_mse": float(train_batch_mse),
    "train_batch_rel_error": float(train_batch_rel_error),
    "validation_loss": float(validation_loss),
    "validation_sample_mse": float(validation_sample_mse),
    "validation_sample_rel_error": float(validation_sample_rel_error),
    "validation_batch_mse": float(validation_batch_mse),
    "validation_batch_rel_error": float(validation_batch_rel_error),
    "residual_mse": float(residual_mse),
    "residual_rel_error": float(residual_rel_error),
    "model_vf_mse": float(model_vf_mse),
    "model_vf_rel_error": float(model_vf_rel_error),
    "states": states,
}

with open("oscillator_wrong_physics_trainable.pkl", "wb") as f:
    pickle.dump(results, f)

visualize_results(results)
