import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import diffrax
import pickle
from jax import grad, jit, vmap
from jax import random
import jax.nn as jnn
from solver import*
from functools import partial



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
print(observed_batch.shape)

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
f_physics_params = jnp.array([0.2,1.0])
params = {"nn_params": nn_params, "f_physics": f_physics_params}
optimizer = optax.multi_transform(
    {
        "train": optax.adam(step_size),
        "freeze": optax.set_to_zero()
    },
    {
        "nn_params": "train",
        "f_physics": "freeze"
    }
)
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

@partial(jit, static_argnames=("step_ratio",))
def update(parameters, opt_state, y0_batch, true_trajectories, h, step_ratio):
  grads = grad(batch_loss)(parameters, y0_batch, true_trajectories, h, step_ratio)
  updates, opt_state = optimizer.update(grads, opt_state, parameters)
  parameters = optax.apply_updates(parameters, updates)
  return parameters, opt_state

best_loss = float("inf")
best_params = params
patience = 500
min_delta = 1e-7
counter = 0

for epoch in range(num_epochs):
    params, opt_state = update(
        params,
        opt_state,
        y0_batch,
        observed_batch,
        h_model,
        ratio
    )

    if epoch % 100 == 0:
        l = batch_loss(
            params,
            y0_batch,
            observed_batch,
            h_model,
            ratio
        )
        print(f"epoch {epoch}, loss = {l}")
        if best_loss - float(l) > min_delta:
            best_loss = float(l)
            best_params = params
            counter = 0
        else:
            counter += 100
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
num_steps = int((observed_batch.shape[1] - 1) * ratio)

pred_full = roll_out(y0_batch[0], h_model, model_rhs, params, num_steps, rk4)
pred_obs = pred_full[::ratio]

## trajectory fitting (position and velocity)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
components = ("x", "v")
for i, ax in enumerate(axes):
    ax.plot(t_obs, observed_batch[index, :, i], "--", alpha=0.7, label="true")
    ax.plot(t_obs, pred_obs[:, i], alpha=0.9, label="predicted")
    ax.set_xlabel("t")
    ax.set_ylabel(components[i])
    ax.set_title(f"Trajectory fitting: {components[i]}")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

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

def plot_quiver(ax, vf, title):
    U = vf[:, 0].reshape(ny, nx)
    W = vf[:, 1].reshape(ny, nx)
    ax.quiver(X, V, U, W, angles="xy")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_quiver(axes[0], true_vf, "true Duffing oscillator")
plot_quiver(axes[1], physics_vf, "physics part")
plot_quiver(axes[2], model_vf, "learned model")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
true_residual = true_vf - physics_vf
plot_quiver(axes[0], true_residual, "true residual (true - physics)")
plot_quiver(axes[1], nn_vf, "learned NN residual")
plt.tight_layout()
plt.show()


# -----------------------------
# Check learned vector field at sample states
# -----------------------------
test_states = jnp.array([
    [-2.0, -1.0],
    [-1.0,  2.0],
    [ 1.0, -2.0],
    [ 2.0,  1.0],
])


# -----------------------------
# Console summary
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
traj_mse = jnp.mean((pred_obs - observed_batch[index]) ** 2)
traj_rel_error = (
    jnp.linalg.norm(pred_obs - observed_batch[index])
    / (jnp.linalg.norm(observed_batch[index]) + eps)
)

print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"experiment type        : oscillator_wrong_physics_trainable")
print(f"ratio                  : {ratio}")
print(f"h_model                : {float(h_model):.6f}")
print(f"best loss              : {best_loss:.6e}")
print(f"learned physics params : {params['f_physics']}")
print(f"true physics params    : [{a}, {-r}]")
print("=" * 80)
print("\nTrajectory check: first 5 observed points")
print("-" * 80)
print(
    f"{'t':>8} | "
    f"{'pred x':>12} {'true x':>12} {'abs err':>10} | "
    f"{'pred v':>12} {'true v':>12} {'abs err':>10}"
)

print("-" * 80)
for i in range(5):
    x_err = abs(float(pred_obs[i, 0] - observed_batch[index, i, 0]))
    v_err = abs(float(pred_obs[i, 1] - observed_batch[index, i, 1]))
    print(
        f"{float(t_obs[i]):8.3f} | "
        f"{float(pred_obs[i,0]):12.6f} "
        f"{float(observed_batch[index,i,0]):12.6f} "
        f"{x_err:10.4e} | "
        f"{float(pred_obs[i,1]):12.6f} "
        f"{float(observed_batch[index,i,1]):12.6f} "
        f"{v_err:10.4e}"
    )

print("\nVector field check at sample states")
print("-" * 125)
print(
    f"{'state':>16} | "
    f"{'NN residual':>24} | "
    f"{'true residual':>24} | "
    f"{'res rel err':>12} | "
    f"{'model rhs':>24} | "
    f"{'true rhs':>24} | "
    f"{'rhs rel err':>12}"
)

print("-" * 125)

for state in test_states:
    nn_res = nn(state, params["nn_params"])
    phys = f_physics(state, params["f_physics"])
    model = model_rhs(state, params)
    true = duffing(0.0, state, None)
    true_res = true - phys
    rhs_rel = jnp.linalg.norm(model - true) / (jnp.linalg.norm(true) + eps)
    res_rel = jnp.linalg.norm(nn_res - true_res) / (jnp.linalg.norm(true_res) + eps)
    print(
        f"{str([float(state[0]), float(state[1])]):>16} | "
        f"{str([round(float(nn_res[0]), 4), round(float(nn_res[1]), 4)]):>24} | "
        f"{str([round(float(true_res[0]), 4), round(float(true_res[1]), 4)]):>24} | "
        f"{float(res_rel):12.4e} | "
        f"{str([round(float(model[0]), 4), round(float(model[1]), 4)]):>24} | "
        f"{str([round(float(true[0]), 4), round(float(true[1]), 4)]):>24} | "
        f"{float(rhs_rel):12.4e}"
    )

print("\nGlobal metrics on vector-field grid")
print("-" * 80)
print(f"trajectory MSE              : {float(traj_mse):.6e}")
print(f"trajectory relative error   : {float(traj_rel_error):.6e}")
print(f"residual MSE                : {float(residual_mse):.6e}")
print(f"residual relative error     : {float(residual_rel_error):.6e}")
print(f"model vector field MSE      : {float(model_vf_mse):.6e}")
print(f"model vector field rel error: {float(model_vf_rel_error):.6e}")
print("=" * 80)

results = {
    "experiment_type": "oscillator_true_physics_fixed",
    "best_loss": float(best_loss),
    "ratio": ratio,
    "h_model": float(h_model),
    "step_size": step_size,
    "pred_traj": pred_obs,
    "true_traj": observed_batch[index],
    "params": params,
    "learned_f_physics": params["f_physics"],
    "true_vf": true_vf,
    "physics_vf": physics_vf,
    "model_vf": model_vf,
    "true_residual": true_residual,
    "learned_residual": nn_vf,
    "traj_mse": float(traj_mse),
    "traj_rel_error": float(traj_rel_error),
    "residual_mse": float(residual_mse),
    "residual_rel_error": float(residual_rel_error),
    "model_vf_mse": float(model_vf_mse),
    "model_vf_rel_error": float(model_vf_rel_error),
    "states": states,
}

with open("oscillator_true_physics_fixed.pkl", "wb") as f:
    pickle.dump(results, f)

