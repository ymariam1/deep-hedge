# ntbn_softplus.py
import torch.nn.functional as fn
from scripts.ntbtn import NoTransactionBandNet  

class SoftplusBandNet(NoTransactionBandNet):
    """NTBN with softplus-positive band widths."""
    def forward(self, input):
        prev_hedge = input[..., [-1]]
        delta      = self.delta(input[..., :-1])
        width      = fn.softplus(self.mlp(input[..., :-1]))

        min_ = delta - width[..., [0]]
        max_ = delta + width[..., [1]]
        return self.clamp(prev_hedge, min=min_, max=max_)
