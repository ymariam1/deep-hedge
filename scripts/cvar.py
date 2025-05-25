# loss_cvar.py
import torch
from torch.nn import Module

class CVaRLoss(Module):
    """
    α-CVaR of negative P&L  +  λ · mean(trades)
    """
    def __init__(self, alpha: float = 0.05, cost_weight: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.cost_weight = cost_weight

    def forward(self, pnl: torch.Tensor, nb_trade: torch.Tensor) -> torch.Tensor:
        # 1) CVaR of losses
        loss_tail = (-pnl).view(-1)                             # losses = -P&L
        q_alpha  = torch.quantile(loss_tail, 1.0 - self.alpha)  # VaR_α
        cvar     = loss_tail[loss_tail >= q_alpha].mean()       # CVaR_α

        # 2) Trade-count regulariser
        trade_penalty = self.cost_weight * nb_trade.float().mean()

        return cvar + trade_penalty
