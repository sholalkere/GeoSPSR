from math import ceil

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm


def bmap(f, x, bs=None, verbose=False, desc=None):
    """
    Batched version of `vmap`
    """
    if not bs:
        bs = x.shape[0]

    n_batch = ceil(x.shape[0] / bs)

    f = vmap(f, 0)
    result = []
    for batch_i in tqdm(range(n_batch), disable=not verbose, desc=desc):
        batch = x[batch_i * bs : (batch_i + 1) * bs]
        result.append(f(batch))

    return jnp.concatenate(result, 0)


def trunc_Zd(n, d=3, flatten=True):
    """
    Returns [-n, n]^d with shape [2n+1, 2n+1, ..., 2n+1, d]
    """
    grid = jnp.mgrid[(slice(-n, n + 1),) * d]
    if flatten:
        return grid.reshape(d, -1).T
    return grid.transpose(list(range(1, d + 1)) + [0])


def interpolator(f, points, bs=None, verbose=False):
    """
    Wrapper for `jax`'s `RegularGridInterpolator`
    """
    dim = len(points)
    grid = jnp.stack(jnp.meshgrid(*points, indexing="ij"), axis=-1)
    values = bmap(
        f, grid.reshape(-1, dim), bs, verbose, desc="Creating interpolator"
    ).reshape(*grid.shape[:-1], -1)

    return RegularGridInterpolator(points, values)


def periodic_stationary_interpolator(
    f, dim, density, bs=128, exponent=1, verbose=False
):
    """
    Create an interpolating object for a periodic stationary kernel with period [0, 2pi]
    Uses grid values of (linspace(-1, 1, density) ** exponent * jnp.pi)^dim
    """
    xis = (jnp.mgrid[-1 : 1 : density * 1j] ** exponent) * jnp.pi
    f_interpolator = interpolator(f, (xis,) * dim, bs, verbose)

    def helper(x1, x2):
        x_diff = x1 - x2
        x_diff = (x_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return f_interpolator(x_diff).squeeze()

    return helper
