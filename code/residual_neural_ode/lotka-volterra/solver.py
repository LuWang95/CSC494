import jax.numpy as jnp
from jax import lax


def solver_step(y, h, rhs, params):
    """
    One numerical step.
    x: current state
    h: step size
    rhs: function, dx/dt = rhs(x,params)
    params: model parameters,as a pytree.
    """

    ...
def forward_euler(y,h,rhs,params):
    d_y = rhs(y,params)
    return y+h*d_y

def heun(y, h, rhs, params):
    d_y_current = rhs(y, params)
    d_y_next = rhs(y + h * d_y_current, params)
    return y + 1/2 * h * (d_y_current + d_y_next)

def rk4(y, h, rhs, params):
    k1 = rhs(y, params)
    k2 = rhs(y + h / 2 * k1, params)
    k3 = rhs(y + h / 2 * k2, params)
    k4 = rhs(y + h * k3, params)
    return y + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def roll_out(y0, h, rhs, params, num_steps, method):
    def step(y, _):
        y_next = method(y, h, rhs, params)
        return y_next,y_next
    _, xs = lax.scan(step, y0, None, length = num_steps)
    return jnp.concatenate([y0[None, :], xs], axis=0)

