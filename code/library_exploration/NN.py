import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from jax import random
from Solver import*


# -----------------------------
# Synthetic data from analytic ODE
# dx/dt = -k x
# x(t) = x0 exp(-kt)
# -----------------------------
k_true = 0.8
x0_true = jnp.array([2.0])   # 1D state
T = 5.0
num_observed = 101
t_obs = jnp.linspace(0.0, T, num_observed)
observed_trajectory = x0_true * jnp.exp(-k_true * t_obs[:, None])
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
num_epochs = 2000
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))
optimizer = optax.adam(step_size)
opt_state = optimizer.init(params)

# def relu(x):
#   return jnp.maximum(x, 0)

def predict(parameters, x):
    activations = x
    for w,b in parameters[:-1]:
        outputs = jnp.dot(w,activations) + b
        activations = jnp.tanh(outputs)
    final_w, final_b = parameters[-1]
    return jnp.dot(final_w, activations) + final_b

batched_predict = vmap(predict,in_axes=(None,0))


def loss(parameters, x0, true_trajectory, h, step_ratio):
    num_steps = (true_trajectory.shape[0] - 1) * step_ratio
    pred_full = roll_out(x0,h,num_steps,parameters,predict)
    pred_useful = pred_full[::step_ratio]
    return jnp.mean((true_trajectory - pred_useful)**2)

def update(parameters, opt_state, x0, true_trajectory, h, step_ratio):
  grads = grad(loss)(parameters, x0, true_trajectory, h, step_ratio)
  updates, opt_state = optimizer.update(grads, opt_state, parameters)
  parameters = optax.apply_updates(parameters, updates)
  return parameters, opt_state






for epoch in range(num_epochs):
    params, opt_state = update(params, opt_state, x0_true, observed_trajectory, h_model, ratio)

    if epoch % 100 == 0:
        l = loss(params, x0_true, observed_trajectory, h_model, ratio)
        print(f"epoch {epoch}, loss = {l}")

# -----------------------------
# Check prediction after training
# -----------------------------
num_steps = (observed_trajectory.shape[0] - 1) * ratio

pred_full = roll_out(x0_true, h_model, num_steps, params, predict)
pred_obs = pred_full[::ratio]

print("pred_obs shape:", pred_obs.shape)
print("observed shape:", observed_trajectory.shape)

print("First 5 predicted values:")
print(pred_obs[:5])

print("First 5 true values:")
print(observed_trajectory[:5])

# -----------------------------
# Check learned vector field f(x)
# true: dx/dt = -0.8x
# -----------------------------
test_x = jnp.array([[0.5], [1.0], [2.0]])

for x in test_x:
    print("x =", x)
    print("NN f(x) =", predict(params, x))
    print("true f(x) =", -k_true * x)
