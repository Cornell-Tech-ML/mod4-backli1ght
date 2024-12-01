from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return float(cls.forward(ctx, *inps))  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition function."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition function."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# mul
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication function."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication function."""
        a, b = ctx.saved_values
        f_a_prime = b
        f_b_prime = a
        return d_output * f_a_prime, d_output * f_b_prime


# inv
class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse function."""
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse function."""
        (a,) = ctx.saved_values
        return -d_output / (a**2)


# neg
class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation function."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation function."""
        return -d_output


# sigmoid
class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid function."""
        result = 1 / (1 + operators.exp(-a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function."""
        (sigmoid_a,) = ctx.saved_values
        return d_output * sigmoid_a * (1 - sigmoid_a)


# relu
class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU function."""
        ctx.save_for_backward(a)
        return max(0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function."""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0


# exp


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential function."""
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential function."""
        (exp_a,) = ctx.saved_values
        return d_output * exp_a


# lt
class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than function."""
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than function."""
        return 0.0, 0.0  # No gradient for comparison


# eq
class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality function."""
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality function."""
        return 0.0, 0.0  # No gradient for comparison
