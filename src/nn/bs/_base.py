from inspect import signature
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import no_type_check
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module

import src.autogreek as autogreek

if TYPE_CHECKING:
    from src.market.derivative.european_option import EuropeanOption


class BSModuleMixin(Module):
    """A mixin class for Black-Scholes formula modules.

    Shape:
        - Input: :math:`(N, *, H_\\text{in})` where
          :math:`H_\\text{in}` is the number of input features and
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Returns delta of the derivative.

        Args:
            input (torch.Tensor): The input tensor. Features are concatenated along
                the last dimension.
                See :meth:`inputs()` for the names of the input features.

        Returns:
            torch.Tensor
        """
        return self.delta(*(input[..., [i]] for i in range(input.size(-1))))

    @no_type_check
    def price(self, *args, **kwargs) -> Tensor:
        """Returns price of the derivative.

        Returns:
            torch.Tensor
        """

    @no_type_check
    def delta(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.delta(self.price, **kwargs)

    @no_type_check
    def gamma(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.gamma(self.price, **kwargs)

    @no_type_check
    def vega(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.vega(self.price, **kwargs)

    @no_type_check
    def theta(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.theta(self.price, **kwargs)

    def inputs(self) -> List[str]:
        """Returns the names of input features.

        Returns:
            list
        """
        return list(signature(self.delta).parameters.keys())


def acquire_params_from_derivative_0(
    derivative: Optional["EuropeanOption"],
    log_moneyness: Optional[Tensor] = None,
    time_to_maturity: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if log_moneyness is None:
        if derivative is None:
            raise ValueError(
                "log_moneyness is required if derivative is not set at this initialization."
            )
        log_moneyness = derivative.log_moneyness()
    if time_to_maturity is None:
        if derivative is None:
            raise ValueError(
                "time_to_maturity is required if derivative is not set at this initialization."
            )
        time_to_maturity = derivative.time_to_maturity()
    return log_moneyness, time_to_maturity


def acquire_params_from_derivative_1(
    derivative: Optional["EuropeanOption"],
    log_moneyness: Optional[Tensor] = None,
    time_to_maturity: Optional[Tensor] = None,
    volatility: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    log_moneyness, time_to_maturity = acquire_params_from_derivative_0(
        derivative=derivative,
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
    )
    if volatility is None:
        if derivative is None:
            raise ValueError(
                "time_to_maturity is required if derivative is not set at this initialization."
            )
        if derivative.ul().volatility is None:
            raise AttributeError(
                "please simulate first and check if volatility exists in the derivative's underlier."
            )
        volatility = derivative.ul().volatility
    return log_moneyness, time_to_maturity, volatility
