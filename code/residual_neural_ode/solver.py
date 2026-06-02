import jax.numpy as jnp
from jax import lax


def solver_step(x, h, rhs, params):
    """
    One numerical step.
    x: current state
    h: step size
    rhs: function, dx/dt = rhs(x,params)
    params: model parameters,as a pytree.
    """

    ...
def forward_euler(x,h,rhs,params):
    d_x = rhs(x,params)
    return x+h*d_x

def roll_out(x0,h,rhs,params,num_steps,method):
    def step(x,_):
        x_next = method(x,h,rhs,params)
        return x_next,x_next
    _, xs = lax.scan(step,x0,None,length = num_steps)
    return jnp.concatenate([x0[None,:], xs], axis=0)

