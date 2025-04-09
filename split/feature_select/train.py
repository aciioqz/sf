import copy
import json
import os
from typing import Dict
from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss, CrossEntropyLoss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from split.splitdataset import get_select_dataset
from t4g.data import BenchDataset
from t4g.data.database import Database
from t4g.data.task import TaskType
from t4g.encoder.text_embedder import GloveTextEmbedding
from t4g.model.graph import (
    get_stype_proposal,
    make_pkey_fkey_graph,
    get_table_index,
)
from t4g.test.relation_batch.model_batch_no_encoder import Model, ModelDataWrapper, get_batch_data_dict


class Args:
    def __init__(self, lr=0.005, epochs=300, dataset=None, task=None, batch_size=128, layers=1):
        self.dataset = dataset
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.channels = 128
        self.aggr = "sum"
        self.num_neighbors = 128
        self.num_workers = 0
        self.layers = layers




def train_candidate_table(seed_value, args, root_dir, raw_file_root_path, json_path):
    seed_everything(seed_value)

    dataset_to_informative_text_cols = {}
    dataset_to_informative_text_cols[args.dataset] = {
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset: BenchDataset = get_select_dataset(name=args.dataset, process=True,
                                        path=raw_file_root_path + "/select/")
    task = dataset.get_task(args.task)

    print("开始除了字段")
    col_to_stype_dict = get_stype_proposal(dataset.db)
    informative_text_cols: Dict = dataset_to_informative_text_cols[args.dataset]
    for table_name, stype_dict in col_to_stype_dict.items():
        for col_name, stype in list(stype_dict.items()):
            # 数据类型中删除label字段
            if table_name == task.entity_table and col_name == task.target_col:
                del stype_dict[col_name]

    cur_base_table = dataset.db.table_dict[task.entity_table].df
    cur_base_table_target_col = cur_base_table[task.target_col].copy(deep=True)

    print(dataset.db.table_dict[task.entity_table].df.columns)
    # 数据类型中删除label字段
    dataset.db.table_dict[task.entity_table].df = dataset.db.table_dict[task.entity_table].df.drop(task.target_col,
                                                                                                   axis=1)
    print(dataset.db.table_dict[task.entity_table].df.columns)
    # 获取target
    print("开始获取target数据")
    temp_index = cur_base_table_target_col.index.to_numpy();
    temp_value = cur_base_table_target_col.values;
    total_length = len(cur_base_table_target_col)
    length_1 = int(total_length * 0.6)
    length_2 = int(total_length * 0.2)
    train_index_arr, train_y = get_id_and_label(temp_index, temp_value, 0, length_1)
    print(type(train_index_arr))
    print(type(train_y))
    val_index_arr, val_y = get_id_and_label(temp_index, temp_value, length_1, length_1 + length_2)
    test_index_arr, test_y = get_id_and_label(temp_index, temp_value, length_1 + length_2, total_length)

    count_1 = np.sum(train_y)
    count_0 = train_y.size - count_1
    print(f"count_0: {count_0}, count_1: {count_1}")

    table_index, table_num, offset_arr = get_table_index(dataset.db, task.entity_table)
    print("Create the graph ")
    data: Data
    data, data_dict = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=device), batch_size=256),
        cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
        table_index=table_index,
        table_num=table_num
    )

    loader_dict: Dict[str, NeighborLoader] = {}
    loader_dict["train"] = NeighborLoader(data, input_nodes=torch.from_numpy(train_index_arr).to(device),
                                          batch_size=args.batch_size, shuffle=True, num_neighbors=[-1, -1])
    loader_dict["val"] = NeighborLoader(data, input_nodes=torch.from_numpy(val_index_arr).to(device),
                                        batch_size=args.batch_size, shuffle=False, num_neighbors=[-1, -1])
    loader_dict["test"] = NeighborLoader(data, input_nodes=torch.from_numpy(test_index_arr).to(device),
                                         batch_size=args.batch_size, shuffle=False, num_neighbors=[-1, -1])
    print("loader_dict")

    print("开始获取index")
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        weights = torch.tensor([count_0 / count_1], dtype=torch.float32).to(device)
        loss_fn = BCEWithLogitsLoss(pos_weight=weights)
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = 15
        loss_fn = CrossEntropyLoss()
        tune_metric = "accuracy"
        higher_is_better = True

    print("开始初始化Model")
    model = Model(data_dict, args, out_channels, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    loss_values = []
    acc_values = []
    state_dict = None
    best_val_metric = 0 if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        print(f"seed: {seed_value}, Epoch: {epoch:02d}")
        train_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict, table_index=table_index,
                                              table_num=table_num, index_arr=train_index_arr,
                                              loader=loader_dict["train"],
                                              y=train_y, device=device, offset_arr=offset_arr)
        train_loss = train_batch(model, optimizer, loss_fn, train_data_wrapper)
        loss_values.append(train_loss)
        val_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict, table_index=table_index,
                                            table_num=table_num, index_arr=val_index_arr, loader=loader_dict["val"],
                                            y=val_y, device=device, offset_arr=offset_arr)
        val_pred, _, _ = test_batch(model, val_data_wrapper, args.batch_size)
        val_metrics = task.evaluate(val_pred, val_y)
        # acc_values.append(val_metrics["accuracy"])

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
        # torch.save(model, f'./model_seed_value_{seed_value}_epoch_{epoch}.pth')
    model.load_state_dict(state_dict)
    val_pred, edge_list, weight_list = test_batch(model, val_data_wrapper, args.batch_size)
    val_metrics = task.evaluate(val_pred, val_y)
    print(f"Best Val metrics: {val_metrics}")
    # plot_loss_with_acc(loss_values, acc_values)

    test_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict, table_index=table_index,
                                         table_num=table_num, index_arr=test_index_arr, loader=loader_dict["test"],
                                         y=test_y, device=device, offset_arr=offset_arr)
    test_pred, edge_list, weights = test_batch_split(model, test_data_wrapper, args.batch_size)
    feature_dict = get_weight(edge_list, weights, table_num, offset_arr, dataset.db.table_dict)
    print(feature_dict)

    for key, value in feature_dict.items():
        feature_dict[key] = float(value)

    with open(json_path, 'w') as json_file:
        json.dump(feature_dict, json_file)

    test_metrics = task.evaluate(test_pred, test_y)
    print(f"Best test metrics: {test_metrics}")


def get_weight(edge_list: List, weight_list: List, table_num, offset_arr, table_dict: Database):
    feature_dict = {}
    feature_count_dict = {}

    top= int(weight_list[0].size(0) * 0.05)
    n_top = 0-top

    for edges, weights in zip(edge_list, weight_list):
        tensor_cpu = weights.cpu().numpy()
        tensor_flat = tensor_cpu.flatten()
        indices = np.argsort(tensor_flat)[n_top:][::-1]
        weight_values = tensor_flat[indices]
        indices = torch.tensor(indices.copy(), dtype=torch.long)
        edges = edges.cpu()
        result = torch.empty((2, top))
        for i in range(edges.shape[0]):
            result[i] = edges[i, indices]

        result = result.T
        for i in range(result.shape[0]):
            first = get_table_num(result[i][0].item(), offset_arr)
            sec = get_table_num(result[i][1].item(), offset_arr)
            if first != sec:
                # print(table_dict[table_num[first]])
                column1 = table_dict[table_num[first]].df.columns.tolist()[1]
                column2 = table_dict[table_num[sec]].df.columns.tolist()[1]
                key = column1 + '&&' + column2
                key2 = column2 + '&&' + column1
                if key in feature_dict or key2 in feature_dict:
                    if key in feature_dict:
                        feature_dict[key] += weight_values[i]
                        feature_count_dict[key] += 1
                    else:
                        feature_dict[key2] += weight_values[i]
                        feature_count_dict[key2] += 1
                else:
                    feature_dict[key] = weight_values[i]
                    feature_count_dict[key] = 1
    #
    # print(feature_dict)
    # print(feature_count_dict)
    # for key, value in feature_count_dict.items():
    #     feature_dict[key] = feature_dict[key] / feature_count_dict[key]

    return feature_dict


def get_table_num(element, offset_arr):
    for i, offset in enumerate(offset_arr):
        if element < offset:
            return i


@torch.no_grad()
def test_batch_split(model, dataWrapper: ModelDataWrapper, batch_size) -> np.ndarray:
    model.eval()
    pred_list = []
    y_list_index = []
    weights_list = []
    edge_list = []
    for batch in dataWrapper.loader:
        batch_data_dict, index_dict, inverse_index_dict = get_batch_data_dict(batch, dataWrapper)
        batch = batch.to(dataWrapper.device)
        x_index = batch.x
        new_index_dict = {k: {val: idx for idx, val in enumerate(v)} for k, v in index_dict.items()}
        pred, edge, weights = model(
            data=batch,
            data_dict=batch_data_dict,
            table_num=dataWrapper.table_num,
            index_dict=new_index_dict,
            inverse_index_dict=inverse_index_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred = torch.sigmoid(pred)
        pred = pred[:batch_size]
        pred_list.append(pred.detach().cpu())
        weights_list.append(weights)
        mapped_edge = x_index[edge]
        edge_list.append(mapped_edge)

    pred = torch.cat(pred_list, dim=0).numpy()
    max_len = len(dataWrapper.y)
    pred = pred[:max_len]

    return pred, edge_list, weights_list


@torch.no_grad()
def test_batch(model, dataWrapper: ModelDataWrapper, batch_size) -> np.ndarray:
    model.eval()
    pred_list = []
    y_list_index = []
    weights_list = []
    edge_list = []
    for batch in dataWrapper.loader:
        batch_data_dict, index_dict, inverse_index_dict = get_batch_data_dict(batch, dataWrapper)
        batch = batch.to(dataWrapper.device)
        x_index = batch.x
        new_index_dict = {k: {val: idx for idx, val in enumerate(v)} for k, v in index_dict.items()}
        pred, edge, weights = model(
            data=batch,
            data_dict=batch_data_dict,
            table_num=dataWrapper.table_num,
            index_dict=new_index_dict,
            inverse_index_dict=inverse_index_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred = torch.sigmoid(pred)
        pred = pred[:batch_size]
        pred_list.append(pred.detach().cpu())
        weights_list.append(weights)
        edge_list.append(edge)

    pred = torch.cat(pred_list, dim=0).numpy()
    max_len = len(dataWrapper.y)
    pred = pred[:max_len]
    return pred, edge_list, weights_list


def train_batch(model, optimizer, loss_fn, dataWrapper):
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = len(dataWrapper.loader)
    for batch in dataWrapper.loader:
        batch_data_dict, index_dict, inverse_index_dict = get_batch_data_dict(batch, dataWrapper)
        batch = batch.to(dataWrapper.device)
        torch.set_printoptions(threshold=np.inf)
        mask = (batch.x >= 0) & (batch.x < len(dataWrapper.y))
        index = torch.nonzero(mask, as_tuple=True)[0]
        optimizer.zero_grad()

        x_index = batch.x
        data_dict = dataWrapper.data_dict
        new_index_dict = {k: {val: idx for idx, val in enumerate(v)} for k, v in index_dict.items()}
        pred, _, _ = model(
            data=batch,
            data_dict=batch_data_dict,
            table_num=dataWrapper.table_num,
            index_dict=new_index_dict,
            inverse_index_dict=inverse_index_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        # 从y中取出对应序号的值
        y = [dataWrapper.y[idx] for idx in batch.x[mask]]
        pred = pred[index]
        loss = loss_fn(pred, torch.tensor(y, dtype=torch.float32).to(dataWrapper.device))
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


def get_id_and_label(index, value, start_l, end_l):
    temp_split_index = index[start_l:end_l]
    temp_split_value = value[start_l:end_l]
    return temp_split_index, temp_split_value


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=(np.array([255, 71, 90]) / 255.))  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()