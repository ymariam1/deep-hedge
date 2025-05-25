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

def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    r"""Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math::

        u(x) = -\exp(-a x) \,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -(-a * input).exp()

def topp(
    input: Tensor, p: float, dim: Optional[int] = None, largest: bool = True
) -> "torch.return_types.return_types.topk":  # type: ignore
    # ToDo(masanorihirano): in torch 1.9.0 or some versions (before 1.13.0), this type and alternatives do not exist)
    """Returns the largest :math:`p * N` elements of the given input tensor,
    where :math:`N` stands for the total number of elements in the input tensor.

    If ``dim`` is not given, the last dimension of the ``input`` is chosen.

    If ``largest`` is ``False`` then the smallest elements are returned.

    A namedtuple of ``(values, indices)`` is returned, where the ``indices``
    are the indices of the elements in the original ``input`` tensor.

    .. seealso::
        - :func:`torch.topk`: Returns the ``k`` largest elements of the given input tensor
          along a given dimension.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Returns:
        Tuple[Tensor, LongTensor] (named tuple)

    Examples:
        >>> from pfhedge.nn.functional import topp
        >>>
        >>> input = torch.arange(1.0, 6.0)
        >>> input
        tensor([1., 2., 3., 4., 5.])
        >>> topp(input, 3 / 5)
        torch.return_types.topk(
        values=tensor([5., 4., 3.]),
        indices=tensor([4, 3, 2]))
    """
    if dim is None:
        return input.topk(ceil(p * input.numel()), largest=largest)
    else:
        return input.topk(ceil(p * input.size(dim)), dim=dim, largest=largest)
    
def expected_shortfall(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import expected_shortfall
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> expected_shortfall(input, 0.3)
        tensor(8.)
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)
    
def isoelastic_utility(input: Tensor, a: float) -> Tensor:
    r"""Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

    .. math::

        u(x) = \begin{cases}
        x^{1 - a} & a \neq 1 \\
        \log{x} & a = 1
        \end{cases} \,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float): Relative risk aversion coefficient of the isoelastic
            utility.

    Returns:
        torch.Tensor
    """
    if a == 1.0:
        return input.log()
    else:
        return input.pow(1.0 - a)

def quadratic_cvar(input: Tensor, lam: float, dim: Optional[int] = None) -> Tensor:
    """Returns the Quadratic CVaR of the given input tensor.

    .. math::

        \\rho (X) = \\inf_\\omega \\left\\{\\omega + \\lambda || \\min\\{0, X + \\omega\\}||_2\\right\\}.

    for :math:`\lambda\geq1`.

    References:
        - Buehler, Hans, Statistical Hedging (March 1, 2019). Available at SSRN: http://dx.doi.org/10.2139/ssrn.2913250
          (See Conclusion.)

    Args:
        input (torch.Tensor): The input tensor.
        lam (float): The :math:`lambda` parameter, representing the weight given to the tail losses.
        dim (int, optional): The dimension to sort along. If None, the tensor is flattened.

    Returns:
        torch.Tensor: The Quadratic CVaR of the input tensor.

    Examples:
        >>> from pfhedge.nn.functional import quadratic_cvar
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> quadratic_cvar(input, 2.0)
        tensor(7.9750)
    """  # NOQA
    if dim is None:
        return quadratic_cvar(input.flatten(), lam, 0)

    output_target = torch.as_tensor(1 / (2 * lam))
    base = input.mean(dim=dim, keepdim=True)
    input = input - base

    def fn_target(omega: Tensor) -> Tensor:
        return fn.relu(-omega - input).mean(dim=dim, keepdim=True)

    lower = torch.amin(-input, dim=dim, keepdim=True) - 1e-8
    upper = torch.amax(-input, dim=dim, keepdim=True) + 1e-8

    precision = 1e-6 * 10 ** int(math.log10((upper - lower).amax()))

    omega = bisect(
        fn=fn_target,
        target=output_target,
        lower=lower,
        upper=upper,
        precision=precision,
    )
    return (
        omega
        + lam * fn.relu(-omega - input).square().mean(dim=dim, keepdim=True)
        - base
    ).squeeze(dim)


def leaky_clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    clamped_slope: float = 0.01,
    inverted_output: str = "mean",
) -> Tensor:
    r"""Leakily clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    See :class:`pfhedge.nn.LeakyClamp` for details.
    """
    x = input

    if min is not None:
        min = torch.as_tensor(min).to(x)
        x = x.maximum(min + clamped_slope * (x - min))

    if max is not None:
        max = torch.as_tensor(max).to(x)
        x = x.minimum(max + clamped_slope * (x - max))

    if min is not None and max is not None:
        if inverted_output == "mean":
            y = (min + max) / 2
        elif inverted_output == "max":
            y = max
        else:
            raise ValueError("inverted_output must be 'mean' or 'max'.")
        x = x.where(min <= max, y)

    return x

def clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    inverted_output: str = "mean",
) -> Tensor:
    r"""Clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    See :class:`pfhedge.nn.Clamp` for details.
    """
    if inverted_output == "mean":
        output = leaky_clamp(input, min, max, clamped_slope=0.0, inverted_output="mean")
    elif inverted_output == "max":
        output = torch.clamp(input, min, max)
    else:
        raise ValueError("inverted_output must be 'mean' or 'max'.")
    return output


def pl(
    spot: Tensor,
    unit: Tensor,
    cost: Optional[List[float]] = None,
    payoff: Optional[Tensor] = None,
    deduct_first_cost: bool = True,
    deduct_final_cost: bool = False,
) -> Tensor:
    r"""Returns the final profit and loss of hedging.

    For
    hedging instruments indexed by :math:`h = 1, \dots, H` and
    time steps :math:`i = 1, \dots, T`,
    the final profit and loss is given by

    .. math::
        \text{PL}(Z, \delta, S) =
            - Z
            + \sum_{h = 1}^{H} \sum_{t = 1}^{T} \left[
                    \delta^{(h)}_{t - 1} (S^{(h)}_{t} - S^{(h)}_{t - 1})
                    - c^{(h)} |\delta^{(h)}_{t} - \delta^{(h)}_{t - 1}| S^{(h)}_{t}
                \right] ,

    where
    :math:`Z` is the payoff of the derivative.
    For each hedging instrument,
    :math:`\{S^{(h)}_t ; t = 1, \dots, T\}` is the spot price,
    :math:`\{\delta^{(h)}_t ; t = 1, \dots, T\}` is the number of shares
    held at each time step.
    We define :math:`\delta^{(h)}_0 = 0` for notational convenience.

    A hedger sells the derivative to its customer and
    obliges to settle the payoff at maturity.
    The dealer hedges the risk of this liability
    by trading the underlying instrument of the derivative.
    The resulting profit and loss is obtained by adding up the payoff to the
    customer, capital gains from the underlying asset, and the transaction cost.

    References:
        - Buehler, H., Gonon, L., Teichmann, J. and Wood, B., 2019.
          Deep hedging. Quantitative Finance, 19(8), pp.1271-1291.
          [arXiv:`1802.03042 <https://arxiv.org/abs/1802.03042>`_ [q-fin]]

    Args:
        spot (torch.Tensor): The spot price of the underlying asset :math:`S`.
        unit (torch.Tensor): The signed number of shares of the underlying asset
            :math:`\delta`.
        cost (list[float], default=None): The proportional transaction cost rate of
            the underlying assets.
        payoff (torch.Tensor, optional): The payoff of the derivative :math:`Z`.
        deduct_first_cost (bool, default=True): Whether to deduct the transaction
            cost of the stock at the first time step.
            If ``False``, :math:`- c |\delta_0| S_1` is omitted the above
            equation of the terminal value.

    Shape:
        - spot: :math:`(N, H, T)` where
          :math:`N` is the number of paths,
          :math:`H` is the number of hedging instruments, and
          :math:`T` is the number of time steps.
        - unit: :math:`(N, H, T)`
        - payoff: :math:`(N)`
        - output: :math:`(N)`.

    Returns:
        torch.Tensor
    """
    # TODO(simaki): Support deduct_final_cost=True
    assert not deduct_final_cost, "not supported"

    if spot.size() != unit.size():
        raise RuntimeError(f"unmatched sizes: spot {spot.size()}, unit {unit.size()}")
    if payoff is not None:
        if payoff.dim() != 1 or spot.size(0) != payoff.size(0):
            raise RuntimeError(
                f"unmatched sizes: spot {spot.size()}, payoff {payoff.size()}"
            )

    output = unit[..., :-1].mul(spot.diff(dim=-1)).sum(dim=(-2, -1))

    if payoff is not None:
        output -= payoff

    if cost is not None:
        c = torch.tensor(cost).to(spot).unsqueeze(0).unsqueeze(-1)
        output -= (spot[..., 1:] * unit.diff(dim=-1).abs() * c).sum(dim=(-2, -1))
        if deduct_first_cost:
            output -= (spot[..., [0]] * unit[..., [0]].abs() * c).sum(dim=(-2, -1))

    return output