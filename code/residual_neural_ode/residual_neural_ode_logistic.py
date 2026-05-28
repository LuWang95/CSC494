import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from jax import random
from solver import*


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
f_physics_params = jnp.array([0.0])
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

def f_physics(x,f_physics_params,):
    return f_physics_params * (x**2)

def model_rhs(x,params):
    return f_physics(x, params["f_physics"]) + nn(x, params["nn_params"])

def loss(parameters, x0, true_trajectory, h, step_ratio):
    num_steps = int((true_trajectory.shape[0] - 1) * step_ratio)
    pred_full = roll_out(x0,h,model_rhs,parameters,num_steps)
    pred_useful = pred_full[::step_ratio]
    return jnp.mean((true_trajectory - pred_useful)**2)

def batch_loss(parameters, x0_batched, true_trajectories, h, step_ratio):
    losses = vmap(loss, in_axes=(None,0,0,None,None))(parameters, x0_batched, true_trajectories, h, step_ratio )
    data_loss = jnp.mean(losses)
    xs = true_trajectories.reshape(-1, 1)
    residuals = vmap(lambda x: nn(x, parameters["nn_params"]))(xs)
    residual_loss = jnp.mean(residuals ** 2)
    lam = 1e-4

    return data_loss + lam * residual_loss

def update(parameters, opt_state, x0_batched, true_trajectories, h, step_ratio):
  grads = grad(batch_loss)(parameters, x0_batched, true_trajectories, h, step_ratio)
  updates, opt_state = optimizer.update(grads, opt_state, parameters)
  parameters = optax.apply_updates(parameters, updates)
  return parameters, opt_state


for epoch in range(num_epochs):
    params, opt_state = update(params, opt_state, x0_batch, observed_batch, h_model, ratio)

    if epoch % 100 == 0:
        l = batch_loss(params, x0_batch, observed_batch, h_model, ratio)
        print(f"epoch {epoch}, loss = {l}")

# -----------------------------
# Check prediction after training
# -----------------------------
index = 0
num_steps = int((observed_batch.shape[1] - 1) * ratio)

pred_full = roll_out(x0_batch[0], h_model, model_rhs, params, num_steps)
pred_obs = pred_full[::ratio]

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