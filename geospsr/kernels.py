from math import prod

from flax import nnx
from jax import numpy as jnp
from jax.numpy import cosh, sinh, tanh


class AbstractKernel(nnx.Module):
    def __init__(self):
        self.dim = 1
        self.variance = 1

    def __call__(self, x1, x2):
        raise NotImplementedError

    def spectral_density(self, n):
        raise NotImplementedError


class Matern32_reference(AbstractKernel):
    """
    Matern 3/2 kernel on S^1 as defined in https://arxiv.org/abs/2006.10160
    """

    def __init__(self, lengthscale=1.0):
        self.lengthscale = lengthscale

    def _helper(self, x1, x2):
        rt3 = 3**0.5
        u = rt3 * (jnp.abs(x1 - x2) - 1 / 2) / self.lengthscale
        result = 2 * self.lengthscale + rt3 / tanh(rt3 / (2 * self.lengthscale))
        result *= jnp.pi**2 * self.lengthscale / 3
        result = result * cosh(u) - 2 * jnp.pi**2 * self.lengthscale**2 / 3 * u * sinh(
            u
        )
        return result

    def __call__(self, x1, x2):
        return self._helper(x1 / (2 * jnp.pi), x2 / (2 * jnp.pi)) / self._helper(0, 0)

    def spectral_density(self, n):
        rt3 = 3**0.5
        result = (3 / (self.lengthscale**2) + 4 * jnp.pi**2 * n**2) ** (-2)
        result *= (
            2
            * rt3
            * sinh(rt3 / (2 * self.lengthscale))
            / ((2 * jnp.pi) ** (-2) * self.lengthscale)
        )
        result /= self._helper(0, 0)
        return result


class Matern32Kernel(AbstractKernel):
    """
    Stable version of Matern 3/2 kernel on S^1 as defined in https://arxiv.org/abs/2006.10160
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def _helper(self, x1, x2):
        rt3 = 3 ** (1 / 2)
        w = rt3 * (jnp.abs(x1 - x2) - (1 / 2))
        w_star = rt3 / 2

        cosh_term = jnp.exp((w - w_star) / self.lengthscale) + jnp.exp(
            (-w - w_star) / self.lengthscale
        )
        first = 2 * self.lengthscale + rt3 / jnp.tanh(rt3 / (2 * self.lengthscale))
        second = 2 * w * jnp.tanh(w / self.lengthscale)

        return cosh_term * (jnp.pi**2 * self.lengthscale / 3) * (first - second)

    def __call__(self, x1, x2):
        return (
            self.variance
            * self._helper(x1 / (2 * jnp.pi), x2 / (2 * jnp.pi))
            / self._helper(0, 0)
        )

    def spectral_density(self, n):
        rt3 = 3**0.5
        second = (3 / (self.lengthscale**2) + 4 * jnp.pi**2 * n**2) ** (-2)
        first = (
            2
            * rt3
            * jnp.tanh(rt3 / (2 * self.lengthscale))
            / ((2 * jnp.pi) ** (-2) * self.lengthscale)
        )

        result = first * second
        result /= self._helper(0, 0)
        return result * self.variance


# https://mathworld.wolfram.com/JacobiThetaFunctions.html
def jtheta3(z, q, trunc_n=50):
    ns = jnp.arange(trunc_n) + 1
    return 1 + 2 * (q ** (ns**2) * jnp.cos(2 * ns * z)).sum()


class SquaredExponentialKernel(AbstractKernel):
    """
    Squared exponential kernel or Matern infinity kernel on S^1 as defined in
    https://arxiv.org/abs/2006.10160

    TODO: Fix stability issues
    """

    def __init__(self, lengthscale, variance):
        self.lengthscale = lengthscale
        self.variance = variance

    def _helper(self, x1, x2):
        return jtheta3(
            jnp.pi * (x1 - x2), jnp.exp(-2 * (jnp.pi * self.lengthscale) ** 2)
        )

    def __call__(self, x1, x2):
        return (
            self.variance
            * self._helper(x1 / (2 * jnp.pi), x2 / (2 * jnp.pi))
            / self._helper(0, 0)
        )

    def spectral_density(self, n):
        exp_inner = -2 * (jnp.pi * self.lengthscale) ** 2
        return (
            self.variance * jnp.exp(exp_inner * n**2) / jtheta3(0, jnp.exp(exp_inner))
        )


class ProductKernel(AbstractKernel):
    """
    Component-wise product of n 1-dimensional kernels
    """

    def __init__(self, *kernels):
        self.kernels = kernels
        self.dim = len(self.kernels)
        self.variance = prod(kernel.variance for kernel in self.kernels)

    def __call__(self, x1, x2):
        result = 1
        for i, kernel in enumerate(self.kernels):
            result *= kernel(x1[i], x2[i])
        return result

    def spectral_density(self, n):
        result = 1
        for i, kernel in enumerate(self.kernels):
            result *= kernel.spectral_density(n[i])
        return result
