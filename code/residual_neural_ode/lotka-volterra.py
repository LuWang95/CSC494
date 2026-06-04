import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import diffrax
from jax import grad, jit, vmap
from jax import random
from solver import*
from functools import partial



# -----------------------------

# Lotka-Volterra model (x = prey, y = predator)
# dx/dt = a x - b x y
# dy/dt = -r y + z x y
# state y = [prey, predator]
# -----------------------------

a = 2.0
b = 0.01
r = 0.005
z = 0.01
y0_batch = jnp.array([
    [15.0, 25.0],
    [10.0, 20.0],
    [20.0, 30.0],
    [12.0, 22.0],
])
T = 5
num_observed = 101
t_obs = jnp.linspace(0.0, T, num_observed)
ratio = 20
h_model = (t_obs[1] - t_obs[0]) / ratio

def lotka_volterra(t, y, args):
    prey, predator = y[0], y[1]
    dxdt = a * prey - b * prey * predator
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
layer_sizes = [state_dim, 32, 32, state_dim]
step_size = 1e-3
num_epochs = 30000
nn_params = init_network_params(layer_sizes, random.key(0))
# incomplete physics: linear growth/decay only (missing xy interaction)
f_physics_params = jnp.array([1.0, -0.01])
params = {"nn_params": nn_params, "f_physics": f_physics_params}
optimizer = optax.adam(step_size)
opt_state = optimizer.init(params)



def nn(y, nn_parameters):
    activations = y
    for w, b in nn_parameters[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jnp.tanh(outputs)
    final_w, final_b = nn_parameters[-1]
    return jnp.dot(final_w, activations) + final_b

## fphysics only learns linear growth/decay
def f_physics(y, f_physics_params):
    prey, predator = y[0], y[1]
    return jnp.array([
        f_physics_params[0] * prey,
        f_physics_params[1] * predator,
    ])

def model_rhs(y, params):
    return f_physics(y, params["f_physics"]) + nn(y, params["nn_params"])

@partial(jit, static_argnames=("step_ratio",))
def loss(parameters, y0, true_trajectory, h, step_ratio):
    num_steps = (true_trajectory.shape[0] - 1) * step_ratio
    pred_full = roll_out(y0,h,model_rhs,parameters,num_steps,forward_euler)
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
    params, opt_state = update(params, opt_state,y0_batch,
                               observed_batch, h_model, ratio)

    l = batch_loss(params, y0_batch, observed_batch, h_model, ratio)
    if best_loss - float(l) > min_delta:
        best_loss = float(l)
        best_params = params
        counter = 0
    else:
        counter += 1
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss = {l}")
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}, best loss = {best_loss}")
        break

params = best_params

# -----------------------------
# Check prediction after training
# -----------------------------
index = 0
num_steps = int((observed_batch.shape[1] - 1) * ratio)

pred_full = roll_out(y0_batch[0], h_model, model_rhs, params, num_steps, forward_euler)
pred_obs = pred_full[::ratio]

## trajectory fitting (both species)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
species = ("prey", "predator")
for i, ax in enumerate(axes):
    ax.plot(t_obs, observed_batch[index, :, i], "--", alpha=0.7, label="true")
    ax.plot(t_obs, pred_obs[:, i], alpha=0.9, label="predicted")
    ax.set_xlabel("t")
    ax.set_ylabel(species[i])
    ax.set_title(f"Trajectory fitting: {species[i]}")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

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
nn_vf = vmap(lambda s: nn(s, params["nn_params"]))(states)
model_vf = vmap(lambda s: model_rhs(s, params))(states)

def plot_quiver(ax, vf, title):
    U = vf[:, 0].reshape(ny, nx)
    V = vf[:, 1].reshape(ny, nx)
    ax.quiver(Prey, Pred, U, V, angles="xy")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_quiver(axes[0], true_vf, "true Lotka-Volterra")
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

print("f_physics params =", params["f_physics"])
print("pred_obs shape:", pred_obs.shape)
print("observed shape:", observed_batch[index].shape)

print("First 5 predicted values:")
print(pred_obs[:5])

print("First 5 true values:")
print(observed_batch[index][:5])

# -----------------------------
# Check learned vector field at sample states
# -----------------------------
test_states = jnp.array([
    [15.0, 25.0],
    [10.0, 20.0],
    [20.0, 30.0],
])

for state in test_states:
    print("state =", state)
    print("NN residual =", nn(state, params["nn_params"]))
    print("f_physics =", f_physics(state, params["f_physics"]))
    print("model rhs =", model_rhs(state, params))
    print("true rhs =", lotka_volterra(0.0, state, None))