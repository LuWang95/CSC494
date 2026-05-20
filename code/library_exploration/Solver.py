from NN import predict
import jax.numpy as jnp
from jax import lax

def forward_euler(x,h,parameters):
    d_x = predict(parameters,x)
    return x+h*d_x

def roll_out(x0,h,num_steps,parameters,):
    def step(x,_):
        x_next = forward_euler(x,h,parameters)
        return x_next,x_next
    _, xs = lax.scan(step,x0,None,num_steps)


    return jnp.concatenate([x0[None,:], xs], axis=0)

