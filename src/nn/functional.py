import math
from math import ceil
from math import pi as kPI
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

from src import autogreek
from src._utils.bisect import bisect
from src._utils.typing import TensorOrScalar


def european_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    """Returns the payoff of a European option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input[..., -1] - strike)
    else:
        return fn.relu(strike - input[..., -1])
    

def european_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    """Returns the payoff of a European binary option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanBinaryOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input[..., -1] >= strike).to(input)
    else:
        return (input[..., -1] <= strike).to(input)
    

def entropic_risk_measure(input: Tensor, a: float = 1.0) -> Tensor:
    """Returns the entropic risk measure.

    See :class:`pfhedge.nn.EntropicRiskMeasure` for details.
    """
    return (torch.logsumexp(-input * a, dim=0) - math.log(input.size(0))) / a


def ncdf(input: Tensor) -> Tensor:
    r"""Returns a new tensor with the normal cumulative distribution function.

    .. math::
        \text{ncdf}(x) =
            \int_{-\infty}^x
            \frac{1}{\sqrt{2 \pi}} e^{-\frac{y^2}{2}} dy

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import ncdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> ncdf(input)
        tensor([0.1587, 0.5000, 1.0000])
    """
    return Normal(0.0, 1.0).cdf(input)

def npdf(input: Tensor) -> Tensor:
    r"""Returns a new tensor with the normal distribution function.

    .. math::
        \text{npdf}(x) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}}

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import npdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> npdf(input)
        tensor([2.4197e-01, 3.9894e-01, 7.6946e-23])
    """
    return Normal(0.0, 1.0).log_prob(input).exp()

def d1(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:
    r"""Returns :math:`d_1` in the Black-Scholes formula.

    .. math::
        d_1 = \frac{s}{\sigma \sqrt{t}} + \frac{\sigma \sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance + variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))

def d2(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:
    r"""Returns :math:`d_2` in the Black-Scholes formula.

    .. math::
        d_2 = \frac{s}{\sigma \sqrt{t}} - \frac{\sigma \sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance - variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def bs_european_price(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
    call: bool = True,
) -> Tensor:
    """Returns Black-Scholes price of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.price`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_price
        ...
        >>> bs_european_price(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([0.0375, 0.0797, 0.1467])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    spot = s.exp() * strike
    price = spot * ncdf(d1(s, t, v)) - strike * ncdf(d2(s, t, v))
    price = price + strike * (1 - s.exp()) if not call else price  # put-call parity

    return price

def bs_european_delta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
) -> Tensor:
    """Returns Black-Scholes delta of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.delta`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_delta
        ...
        >>> bs_european_delta(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([0.3446, 0.5398, 0.7257])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    delta = ncdf(d1(s, t, v))
    delta = delta - 1 if not call else delta  # put-call parity

    return delta


def bs_european_gamma(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes gamma of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.gamma`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_gamma
        ...
        >>> bs_european_gamma(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([2.0350, 1.9848, 1.5076])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = strike * s.exp()
    numerator = npdf(d1(s, t, v))
    denominator = spot * v * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )


def bs_european_vega(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes vega of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.vega` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    return npdf(d1(s, t, v)) * price * t.sqrt()


def bs_european_theta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes theta of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.theta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    numerator = -npdf(d1(s, t, v)) * price * v
    denominator = 2 * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )