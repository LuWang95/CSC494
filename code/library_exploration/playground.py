import jax.numpy as jnp
import numpy as np
import time
from jax import jit

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)

norm_compiled = jit(norm)

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
print(np.allclose(norm(X), norm_compiled(X), atol=1E-6))


norm_compiled(X).block_until_ready() ##must compile before the benchmark
start = time.time()
norm(X).block_until_ready()
end = time.time()
print("normal:", end - start)
start = time.time()
norm_compiled(X).block_until_ready()
end = time.time()
print("jit:", end - start)