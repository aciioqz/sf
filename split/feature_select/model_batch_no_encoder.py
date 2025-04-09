from typing import Dict, List, Any

import torch
import torch_frame
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.stats import StatType
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from t4g.encoder.encoder import MyTransformer
from t4g.model.graph import TableData
from t4g.model.nn import GraphSAGE


class Model(torch.nn.Module):
    def __init__(self, data_dict, args, out_channels, layers):
        super().__init__()

        self.encoder = Encoder(
            channels=args.channels,
            node_to_col_names_dict={
                node_type: data_dict[node_type].tf.col_names_dict
                for node_type in data_dict
            },
            node_to_col_stats={
                node_type: data_dict[node_type].col_stats
                for node_type in data_dict
            },
        )

        self.gnn = GraphSAGE(
            in_channels=args.channels,
            hidden_channels=args.channels,
            aggr=args.aggr,
            num_layers=layers
        )
        self.head = MLP(
            args.channels,
            out_channels=out_channels,
            num_layers=1,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
            self,
            data: Data,
            data_dict,
            table_num,
            index_dict,
            inverse_index_dict,
    ) -> Tensor:
        x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict={
            node_type: data_dict[node_type].tf
            for node_type in data_dict
        })

        tensors_to_concat = []

        data_x = data.x.tolist()
        for b in data_x:
            table_index = inverse_index_dict[b]
            table_name = table_num[table_index]
            tuple_index = index_dict[table_index][b]
            temp_tuple_tensor = x_dict[table_name][tuple_index]
            tensors_to_concat.append(temp_tuple_tensor)
            # tensors_to_concat = torch.cat((tensors_to_concat, temp_tuple_tensor), dim=0)
        tensors_to_concat = torch.stack(tensors_to_concat, dim=0)
        x_dict = tensors_to_concat
        x_dict = self.gnn(
            x_dict,
            data.edge_index,
        )
        head = self.head(x_dict)

        return head


class ModelDataWrapper:
    def __init__(self, data, data_dict, table_index, table_num, index_arr, loader, y, device, offset_arr):
        """
        data_dict = 'base': <t4g.model.graph.TableData object at 0x7fb6b11aa490>, 'kr_array2': <t4g.model.graph.TableData object at 0x7fb6b11aac10>, 'kr_array5': <t4g.model.graph.TableData object at 0x7fb6b11aadc0>,
        offset_arr = [1000, 2000, 3000]
        table_num = {0: 'base', 1: 'kr_array2', 2: 'kr_array5'}
        """
        self.data = data
        self.data_dict = data_dict
        self.table_index = table_index
        self.table_num = table_num
        self.index_arr = index_arr
        self.loader = loader
        self.y = y
        self.device = device
        self.offset_arr = offset_arr



def get_batch_data_dict(batch, dataWrapper):
    # index_dict = {0: [6, 4, 5], 1: [12]}
    index_dict, inverse_index_dict = find_indices(dataWrapper.offset_arr, batch.x.tolist())
    data_dict = {}
    for key, value in index_dict.items():
        # offset_arr = [1000, 2000, 3000]
        offset = dataWrapper.offset_arr[key]
        # table_num = {0: 'base', 1: 'kr_array2', 2: 'kr_array5'}
        table_name = dataWrapper.table_num[key]
        """
        table_index = {'base': {0: 0, 1: 1,..., 999:999}, 
            'kr_array2': {0: 1000, 1: 1001,..., 999: 1999}, 
            'kr_array5': {0: 2000, 1: 2001,..., 999: 2999}
        """
        # {0: 1000, 1: 1001,..., 999: 1999} key is id
        tuple_id_dict = dataWrapper.table_index[table_name]
        # reverse {1000:0, 1001: 1,..., 1999: 999} to find id
        inverted_tuple_dict = {value: key for key, value in tuple_id_dict.items()}
        id_list = []
        for v in value:
            id_list.append(inverted_tuple_dict[v])
        # id_list.sort()
        table_data = dataWrapper.data_dict[table_name]

        data_dict[table_name] = TableData()
        new_tf = extract_rows(table_data, id_list)
        data_dict[table_name].tf = new_tf.to(dataWrapper.device)
        data_dict[table_name].col_stats = table_data.col_stats

    return data_dict, index_dict, inverse_index_dict


def extract_rows(data: TableData, indices) -> TensorFrame:
    # indices: index list
    # 定义新的字典用于存储提取后的特征
    new_feat_dict = {}

    # 遍历原始特征字典
    for key, tensor in data.tf.feat_dict.items():
        if isinstance(tensor, torch.Tensor):
            # 提取指定的行
            new_feat_dict[key] = tensor[indices]
        elif isinstance(tensor, MultiEmbeddingTensor):
            new_values = tensor.values[indices, :]
            new_num_rows = len(indices)
            # 更新 num_rows
            # 返回新的 MultiEmbeddingTensor 实例
            new_feat_dict[key] = MultiEmbeddingTensor(
                num_rows=new_num_rows,
                num_cols=tensor.num_cols,
                values=new_values,
                offset=tensor.offset
            )
        else:
            raise ValueError(f'Unsupported tensor type: {type(tensor)} for key: {key}')

    # 创建新的 TensorFrame
    new_tf = TensorFrame(
        feat_dict=new_feat_dict,
        col_names_dict=data.tf.col_names_dict,
        y=data.tf.y[indices] if data.tf.y is not None else None
    )

    return new_tf

def find_indices(arr, batch_x):
    result_dict = {}
    inverse_dict = {}
    for b in batch_x:
        for i, a in enumerate(arr):
            if b < a:
                if i not in result_dict:
                    result_dict[i] = []
                result_dict[i].append(b)
                inverse_dict[b] = i
                break
    return result_dict, inverse_dict

class Encoder(torch.nn.Module):

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

if __name__ == '__main__':
    A = torch.tensor([7, 8, 9])
    B = torch.tensor([2, 0, 1])
    print(A[B])
