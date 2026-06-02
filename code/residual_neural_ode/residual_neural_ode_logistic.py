import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax import random
from solver import*
from functools import partial


def generate_true_trajectory(t,x0):
    return 1.0 / (1.0 + ((1.0 - x0) / x0) * jnp.exp(-t))

# -----------------------------
# Synthetic data from analytic ODE
# dx/dt = x(1-x) x0 = 1/2
# x(t) = exp(t)/1+exp(t)
# -----------------------------
k_true = -1
x0_batch = jnp.array([
    [0.2], [0.4], [0.6], [0.8],
    [1.2], [1.5], [2.0], [2.5]
])
T = 2.0
num_observed = 101
t_obs = jnp.linspace(0.0, T, num_observed)
observed_batch = vmap(lambda x0: generate_true_trajectory(t_obs[:,None], x0))(x0_batch)
# observed_batch = vmap(generate_true_trajectory,in_axes=(None,0))(t_obs[:,],x0_batch) alternate syntax
ratio = 10
h_model = (t_obs[1] - t_obs[0]) / ratio

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes)-1)
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


layer_sizes = [1, 32, 32, 1]
step_size = 1e-2
num_epochs = 5000
nn_params = init_network_params(layer_sizes, random.key(0))
f_physics_params = jnp.array([-0.1])
params = {"nn_params": nn_params, "f_physics": f_physics_params}
optimizer = optax.adam(step_size)
opt_state = optimizer.init(params)



def nn(x,nn_parameters):
    activations = x
    for w,b in nn_parameters[:-1]:
        outputs = jnp.dot(w,activations) + b
        activations = jnp.tanh(outputs)
    final_w, final_b = nn_parameters[-1]
    return jnp.dot(final_w, activations) + final_b

def f_physics(x,f_physics_params):
    return f_physics_params * (x**2)

def model_rhs(x,params):
    return f_physics(x, params["f_physics"]) + nn(x, params["nn_params"])

@partial(jit, static_argnames=("step_ratio",))
def loss(parameters, x0, true_trajectory, h, step_ratio):
    num_steps = (true_trajectory.shape[0] - 1) * step_ratio
    pred_full = roll_out(x0,h,model_rhs,parameters,num_steps,forward_euler)
    pred_useful = pred_full[::step_ratio]
    return jnp.mean((true_trajectory - pred_useful)**2)


@partial(jit, static_argnames=("step_ratio",))
def batch_loss(parameters, x0_batched, true_trajectories, h, step_ratio):
    losses = vmap(loss, in_axes=(None,0,0,None,None))(parameters, x0_batched, true_trajectories, h, step_ratio )
    data_loss = jnp.mean(losses)
    return data_loss

@partial(jit, static_argnames=("step_ratio",))
def update(parameters, opt_state, x0_batched, true_trajectories, h, step_ratio):
  grads = grad(batch_loss)(parameters, x0_batched, true_trajectories, h, step_ratio)
  updates, opt_state = optimizer.update(grads, opt_state, parameters)
  parameters = optax.apply_updates(parameters, updates)
  return parameters, opt_state

best_loss = float("inf")
best_params = params
patience = 500
min_delta = 1e-7
counter = 0

for epoch in range(num_epochs):
    params, opt_state = update(params, opt_state,x0_batch,
                               observed_batch, h_model, ratio)

    l = batch_loss(params, x0_batch, observed_batch, h_model, ratio)
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

pred_full = roll_out(x0_batch[0], h_model, model_rhs, params, num_steps, forward_euler)
pred_obs = pred_full[::ratio]

## trajectory fitting
plt.figure(figsize=(10,5))
plt.plot(t_obs, observed_batch[index, :, 0], "--", alpha=0.7)
plt.plot(t_obs, pred_obs[:, 0], alpha=0.9)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Trajectory fitting: true vs predicted")
plt.grid(True)
plt.show()

## vector field
x_grid = jnp.linspace(0.0, 2.8, 200)[:, None]
true_f = x_grid * (1 - x_grid)
true_physics = - x_grid * x_grid
physics_f = f_physics(x_grid, params["f_physics"])
nn_f = vmap(lambda x: nn(x, params["nn_params"]))(x_grid)
model_f = physics_f + nn_f
plt.figure(figsize=(7, 4))
plt.plot(x_grid[:, 0], true_f[:, 0], label="true f(x)=x(1-x)")
plt.plot(x_grid[:, 0], model_f[:, 0], label="learned model f(x)")
plt.plot(x_grid[:, 0], physics_f[:, 0], label="physics part")
plt.plot(x_grid[:, 0], nn_f[:, 0], label="NN residual")
plt.axhline(0, color = 'black', linewidth=1)
plt.xlabel("x")
plt.ylabel("dx/dt")
plt.title("Learned vector field")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Visualization 3: residual target
# true residual = f_true - f_physics
# -----------------------------
true_residual = true_f - physics_f
plt.figure(figsize=(7, 4))
plt.plot(x_grid[:, 0], true_residual[:, 0], label="true residual")
plt.plot(x_grid[:, 0], nn_f[:, 0], label="learned NN residual")
plt.axhline(0, color = 'black',linewidth=1)
plt.xlabel("x")
plt.ylabel("residual")
plt.title("Residual: true missing physics vs learned NN")
plt.legend()
plt.grid(True)
plt.show()


print("k =",params["f_physics"])
print("pred_obs shape:", pred_obs.shape)
print("observed shape:", observed_batch[index].shape)

print("First 5 predicted values:")
print(pred_obs[:5])

print("First 5 true values:")
print(observed_batch[index][:5])

# -----------------------------
# Check learned vector field f(x)
# true: dx/dt = x(1-x)
# -----------------------------
test_x = jnp.array([[0.5], [0.7], [0.9], [1.0]])

for x in test_x:
    print("x =", x)
    print("NN f(x) =", nn(x, params["nn_params"]))
    print("f_physics(x) =", f_physics(x, params["f_physics"]))
    print("model f(x) =", model_rhs(x, params))
    print("true f(x) =", x * (1 - x))