import os
from typing import Any, Dict, NamedTuple, Optional

import numpy
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from t4g.util.data_type import infer_df_stype
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index

from t4g.data import Database, Table
from t4g.data.task import Task
from t4g.util.table_knn import table_encoder_list, convert_input
from t4g.util.utils import find_k_nearest_neighbors_list, find_top_similar_rows
import csv


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    return torch.from_numpy(ser.astype(int).values) // 10 ** 9


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, Any]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): : The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.table_dict.items():
        # Take the first 10,000 rows for quick stype inference.
        inferred_col_to_stype = infer_df_stype(table.df)

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        if table.pkey_col is not None:
            inferred_col_to_stype.pop(table.pkey_col)
        for fkey in table.fkey_col_to_pkey_table.keys():
            inferred_col_to_stype.pop(fkey)

        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype

    return inferred_col_to_stype_dict


def make_pkey_fkey_graph(
        db: Database,
        col_to_stype_dict: Dict[str, Dict[str, stype]],
        text_embedder_cfg: Optional[TextEmbedderConfig] = None,
        cache_dir: Optional[str] = None,
        table_index: Dict[str, Dict[int, int]] = None,
        table_num: Dict[int, str] = None,
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print("make_pkey_fkey_graph")
    print(device)
    data = Data()
    data_dict = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    index_x = []
    cur = 0

    tensor_list = []
    for key in sorted(table_num.keys()):
        table_name = table_num[key]
        print(f'table_name = {table_name}')
        table = db.table_dict[table_name]
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            df = pd.DataFrame({"__const__": np.ones(len(table.df))})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        df = table.df
        if table_name not in data_dict:
            data_dict[table_name] = TableData()

        data_dict[table_name].tf = dataset.tensor_frame
        data_dict[table_name].col_stats = dataset.col_stats
        ptable_index_map = table_index[table_name]

        if table.fkey_col_to_pkey_table is None:
            continue
        for temp_index in df[table.pkey_col]:
            index_x.append(cur)
            cur = cur + 1
        print("add edge")
        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            ftable_index_map = table_index[pkey_table_name]

            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            pkey_index_n = torch.tensor([ptable_index_map[item.item()] for item in fkey_index])
            fkey_index_n = torch.tensor([ftable_index_map[item.item()] for item in pkey_index])

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index_n, pkey_index_n], dim=0)
            # edge = sort_edge_index(edge_index)
            tensor_list.append(edge_index)

            # pkey -> fkey edges
            edge_index2 = torch.stack([pkey_index_n, fkey_index_n], dim=0)
            # edge2 = sort_edge_index(edge_index2)
            tensor_list.append(edge_index2)

    print('add_edge_end')

    all_edge = torch.cat(tensor_list, dim=1)
    data.edge_index = all_edge


    # join_edge_arr = np.ones(all_edge, dtype=int)
    # edge_arr = join_edge_arr.tolist()
    # data.edge_attr = torch.tensor(edge_arr)
    data.x = torch.tensor(index_x)
    data.validate()

    return data, data_dict


def make_pkey_fkey_graph_knn2(
        db: Database,
        col_to_stype_dict: Dict[str, Dict[str, stype]],
        text_embedder_cfg: Optional[TextEmbedderConfig] = None,
        cache_dir: Optional[str] = None,
        table_index: Dict[str, Dict[int, int]] = None,
        table_num: Dict[int, str] = None,
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print("make_pkey_fkey_graph")
    print(device)
    data = Data()
    data_dict = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    index_x = []
    cur = 0

    base_table_name = table_num[0]
    print(f'base_table_name = {base_table_name}')
    base_table = db.table_dict[base_table_name]
    base_dataset = []

    tensor_list = []
    similar_edge_list = []
    similar_arr_list = []
    for key in sorted(table_num.keys()):
        table_name = table_num[key]
        print(f'table_name = {table_name}')
        table: Table = db.table_dict[table_name]
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            df = pd.DataFrame({"__const__": np.ones(len(table.df))})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)
        if base_table_name in table_name:
            base_dataset = dataset

        df = table.df
        if table_name not in data_dict:
            data_dict[table_name] = TableData()

        data_dict[table_name].tf = dataset.tensor_frame
        data_dict[table_name].col_stats = dataset.col_stats
        ptable_index_map = table_index[table_name]

        if table.fkey_col_to_pkey_table is None:
            continue
        for temp_index in df[table.pkey_col]:
            index_x.append(cur)
            cur = cur + 1

        if table.is_need_edge == True:
            print(f'add {table_name} similarity edges')
            # encoder = table_encoder_list(base_dataset)
            encoder = convert_input(dataset)
            # tensor_stack = torch.stack(encoder)
            # neighbors_list, similar_arr = find_k_nearest_neighbors_list(encoder, 10)
            neighbors_list, similar_arr =find_top_similar_rows(tensor = encoder)
            neighbors_list = [[ptable_index_map[elem] for elem in sublist] for sublist in neighbors_list]
            if len(neighbors_list)>0 :
                knn_edge_tensor = torch.tensor(neighbors_list).transpose(0, 1)
                similar_edge_list.append(knn_edge_tensor)
                similar_arr_list.extend(similar_arr)
                print('add similarity edges end !')
            else:
                print('add similarity 0 edges end !')

        print(f"add {table_name} joinable edges")
        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            ftable_index_map = table_index[pkey_table_name]

            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            pkey_index_n = torch.tensor([ptable_index_map[item.item()] for item in fkey_index])
            fkey_index_n = torch.tensor([ftable_index_map[item.item()] for item in pkey_index])

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index_n, pkey_index_n], dim=0)
            # edge = sort_edge_index(edge_index)
            tensor_list.append(edge_index)

            # pkey -> fkey edges
            edge_index2 = torch.stack([pkey_index_n, fkey_index_n], dim=0)
            # edge2 = sort_edge_index(edge_index2)
            tensor_list.append(edge_index2)

    all_edge = torch.cat(tensor_list, dim=1)
    join_edge_size = all_edge.shape[1]
    join_edge_arr = np.ones(join_edge_size, dtype=int)

    similar_edge = torch.cat(similar_edge_list, dim=1)

    edge_arr = join_edge_arr.tolist() + similar_arr_list
    data.edge_attr = torch.tensor(edge_arr)

    edges = torch.cat((all_edge, similar_edge), dim=1)
    data.edge_index = edges
    data.x = torch.tensor(index_x)
    data.validate()

    return data, data_dict


def get_similar_edge(table_dict: Dict[str, Table], base_dataset: Database):
    print("get_similar_edge")


def get_table_index(db: Database, entity_table: str):
    r""".Give each row in each table a unique encoding.
        Encoding starts from the Base table and starts from 0.

        Args:
            db (Database): A database object containing a set of tables.
            col_to_stype_dict (Dict[str, Dict[str, stype]]): Column to stype for
                each table.
            entity_table (str): Base table.

        Returns:
           Dict[str, Dict[int, int]]，The list only contains two elements, the starting and ending positions.
        """
    table_index = {}
    table_num = {}
    offset_arr = []
    table_dict = db.table_dict
    table_base = table_dict[entity_table]
    if table_base is None:
        raise ValueError(
            f"base table {table_base} must not be empty."
        )
    cur = 0
    num = 0
    table_num[num] = entity_table
    cur_index, cur = single_table_index(cur=cur, cur_table=table_base, db=db)
    num = num + 1
    table_index[entity_table] = cur_index
    offset_arr.append(cur)
    for table_name, table in db.table_dict.items():
        if table_name == entity_table:
            continue
        table_num[num] = table_name
        temp_index, cur = single_table_index(cur=cur, cur_table=table, db=db)
        table_index[table_name] = temp_index
        num = num + 1
        offset_arr.append(cur)
    return table_index, table_num, offset_arr


def single_table_index(cur: int, cur_table: Table, db: Database):
    index_dict = {}
    df = cur_table.df
    # Ensure that pkey is consecutive.
    if cur_table.pkey_col is not None:
        assert (df[cur_table.pkey_col].values == np.arange(len(df))).all()
    pkey_index = df[cur_table.pkey_col]
    for index, v in enumerate(pkey_index):
        index_dict[v] = cur
        cur = cur + 1
    return index_dict, cur


class AttachTargetTransform:
    r"""Adds the target label to the heterogeneous mini-batch.
    The batch consists of disjoins subgraphs loaded via temporal sampling.
    The same input node can occur twice with different timestamps, and thus
    different subgraphs and labels. Hence labels cannot be stored in the graph
    object directly, and must be attached to the batch after the batch is
    created."""

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: Data) -> Data:
        batch.y = self.target
        return batch


class TrainTableInput(NamedTuple):
    nodes: Tensor
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]
    entity_table: str


def get_train_table_input(
        table: Table,
        task: Task,
) -> TrainTableInput:
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = to_unix_time(table.df[table.time_col])

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == "multiclass_classification":
            target_type = int
        target = torch.from_numpy(table.df[task.target_col].values.astype(target_type))
        transform = AttachTargetTransform(task.entity_table, target)

    return TrainTableInput(
        nodes=nodes,
        time=time,
        target=target,
        transform=transform,
        entity_table=task.entity_table,
    )


class TableData:
    def __init__(self):
        self.tf = None
        self.col_stats = None
        self.time = None


def find_k_hops(hops, unique_nodes, edge_index):
    res = []
    new_index = edge_index
    for _ in range(hops):
        res, new_index = find_hops(unique_nodes, new_index)
        unique_nodes = res
    return res, new_index

def find_hops(unique_nodes, edge_index):
    res = []
    arr_set = set(unique_nodes)
    new_edge_index = []

    for i in range(len(edge_index[0])):
        num1 = edge_index[0][i].item()
        num2 = edge_index[1][i].item()

        if num1 in arr_set:
            res.append(num2)
        elif num2 in arr_set:
            res.append(num1)
        elif [num1, num2] not in new_edge_index:
            new_edge_index.append([num1, num2])

    res = set(res)
    return res, torch.tensor(new_edge_index).transpose(0, 1)


if __name__ == '__main__':
    input_list = numpy.array([2])
    edge_index = torch.tensor([[1, 2, 3, 4, 5],
                               [5, 4, 4, 2, 1]])
    res, new_index = find_k_hops(3, input_list, edge_index)
    print(res)
    print(new_index)
