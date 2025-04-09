from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet, TabTransformer
from t4g.encoder.encoder import MyTransformer
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv, GATConv, \
    GCNConv, GINConv, SGConv, GatedGraphConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer
import torch.nn.functional as F


class HeteroEncoder(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
            node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
            torch_frame_model_cls=MyTransformer,
            torch_frame_model_kwargs: Dict[str, Any] = {
                "channels": 128,
                "num_layers": 4,
            },
            default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
                torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
                torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
                torch_frame.multicategorical: (
                        torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                        {},
                ),
                torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
                torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
            },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
            self,
            tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroGraph(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            conv = HeteroConv(
                {
                    edge_type: GATConv((channels, channels), channels, heads=heads, add_self_loops=False, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class HeteroGraphGCN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GCNConv(channels, channels, add_self_loops=False)
                    # K is the number of propagation steps
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            conv = HeteroConv(
                {
                    edge_type: GATConv((channels, channels), channels, heads=heads, add_self_loops=False, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for step, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # x_dict = conv(x_dict, edge_index_dict)
            if step % 2 == 0:
                x_dict = conv(x_dict, edge_index_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphGIN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GINConv(
                        torch.nn.Sequential(
                            torch.nn.Linear(channels, channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(channels, channels),
                        ), aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            conv = HeteroConv(
                {
                    edge_type: GATConv((channels, channels), channels, heads=heads, add_self_loops=False, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphGGNN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        conv = HeteroConv(
            {
                edge_type: GatedGraphConv(channels, num_layers=num_layers, aggr=aggr)
                for edge_type in edge_types
            },
            aggr=aggr,
        )
        self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphNoWeight(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


# class HeteroGraphSAGE(torch.nn.Module):
#     def __init__(
#             self,
#             node_types: List[NodeType],
#             edge_types: List[EdgeType],
#             channels: int,
#             aggr: str = "mean",
#             num_layers: int = 2,
#     ):
#         super().__init__()
#
#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv(
#                 {
#                     edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
#                     for edge_type in edge_types
#                 },
#                 aggr="sum",
#             )
#             self.convs.append(conv)
#
#         self.norms = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             norm_dict = torch.nn.ModuleDict()
#             for node_type in node_types:
#                 norm_dict[node_type] = LayerNorm(channels, mode="node")
#             self.norms.append(norm_dict)
#         self.dropout = 0.3
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for norm_dict in self.norms:
#             for norm in norm_dict.values():
#                 norm.reset_parameters()
#
#     def forward(
#             self,
#             x_dict: Dict[NodeType, Tensor],
#             edge_index_dict: Dict[NodeType, Tensor],
#             num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
#             num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
#     ) -> Dict[NodeType, Tensor]:
#         for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
#             # # Trim graph and features to only hold required data per layer:
#             # if num_sampled_nodes_dict is not None:
#             #     assert num_sampled_edges_dict is not None
#             #     x_dict, edge_index_dict, _ = trim_to_layer(
#             #         layer=i,
#             #         num_sampled_nodes_per_hop=num_sampled_nodes_dict,
#             #         num_sampled_edges_per_hop=num_sampled_edges_dict,
#             #         x=x_dict,
#             #         edge_index=edge_index_dict,
#             #     )
#
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
#             x_dict = {key: x.relu() for key, x in x_dict.items()}
#             # x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}  # 添加 dropout
#         return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "mean",
            num_layers: int = 1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Trim graph and features to only hold required data per layer:
            if num_sampled_nodes_dict is not None:
                assert num_sampled_edges_dict is not None
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                    num_sampled_edges_per_hop=num_sampled_edges_dict,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
