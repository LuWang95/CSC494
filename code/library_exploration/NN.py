import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# we assume this is for the predator-prey model, so the dimension of input_layer is 2：
layer_sizes = [2, 512, 512, 1]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))

def relu(x):
  return jnp.maximum(x, 0)

def predict(parameters, x):
    activations = x
    for w,b in parameters[:-1]:
        outputs = jnp.dot(w,activations) + b
        activations = relu(outputs)
    final_w, final_b = parameters[-1]
    return jnp.dot(final_w, activations) + final_b

batched_predict = vmap(predict,in_axes=(0, None, 0, None))

