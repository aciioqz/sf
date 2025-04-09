from torch import Tensor
import math
from typing import Any, Union, Dict, List
import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.nn.conv import TabTransformerConv
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
    StypeEncoder,
)
from torch_frame.data.stats import StatType
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
    ModuleList,
)

from torch_frame.nn.conv import TableConv
from torch_frame.nn.conv.tab_transformer_conv import SelfAttention

class BaselineEncoder(Module):
    def __init__(
            self,
            channels: int,
            out_channels: int,
            num_layers: int,
            col_stats: Dict[str, Dict[StatType, Any]],
            col_names_dict: Dict[torch_frame.stype, List[str]],
            stype_encoder_dict: Union[Dict[torch_frame.stype, StypeEncoder], None] = None,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),

            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.decoder = Linear(channels, out_channels)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        out = self.decoder(x.mean(dim=1))
        return out


class MyTransformer(Module):
    def __init__(
            self,
            channels: int,
            out_channels: int,
            num_layers: int,
            col_stats: Dict[str, Dict[StatType, Any]],
            col_names_dict: Dict[torch_frame.stype, List[str]],
            stype_encoder_dict: Union[Dict[torch_frame.stype, StypeEncoder], None] = None,
            dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),

            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        self.convs = ModuleList([
            TransformerConv(
                channels=channels,
                num_heads=8,
                attn_dropout = dropout_prob
            ) for _ in range(num_layers)
        ])
        self.decoder = Linear(channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        for conv in self.convs:
            x = conv(x)
        out = self.decoder(x.mean(dim=1))
        return out


class TransformerConv(TableConv):
    r"""
    Args:
        channels (int): Input/output channel dimensionality
        num_heads (int): Number of attention heads
        attn_dropout (float): attention module dropout (default: :obj:`0.`)
    """

    def __init__(self, channels: int,
                 num_heads: int,
                 attn_dropout: float = 0.):
        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.attn = SelfAttention(channels, num_heads, attn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        out = self.attn(x)
        x = x + out
        return x

    def reset_parameters(self):
        self.norm_1.reset_parameters()
        self.attn.reset_parameters()