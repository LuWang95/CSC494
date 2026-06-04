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

def heun(x,h,rhs,params):
    d_x_current = rhs(x,params)
    d_x_next = rhs(x + h*d_x_current,params)
    return x + 1/2 * h * (d_x_current + d_x_next)

def rk4(x, h, rhs, params):
    k1 = rhs(x, params)
    k2 = rhs(x + h/2 * k1, params)
    k3 = rhs(x + h/2 * k2, params)
    k4 = rhs(x + h * k3, params)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def roll_out(x0,h,rhs,params,num_steps,method):
    def step(x,_):
        x_next = method(x,h,rhs,params)
        return x_next,x_next
    _, xs = lax.scan(step,x0,None,length = num_steps)
    return jnp.concatenate([x0[None,:], xs], axis=0)

