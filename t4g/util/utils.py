import shutil
from pathlib import Path
from typing import Dict
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pooch
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import numpy as np


def to_unix_time(column: pd.Series) -> pd.Series:
    times = pd.to_datetime(column, errors='coerce')
    """convert a timestamp column to unix time"""
    result = pd.to_datetime(column).astype('int64') // 10**9 // 3600
    result[times.isnull()] = 0
    return result
    # return pd.to_datetime(column).astype("datetime64[s]")


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    shutil.unpack_archive(zip_path, unzip_path)
    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
        ]


def read_csv_to_dict(path: str, key_column: str, value_column: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    values = df[value_column].values
    keys = df[key_column]
    if len(values) != len(keys):
        raise ValueError("数组长度不匹配")

    my_dict = {key: value for key, value in zip(keys, values)}

    return my_dict


import torch


def find_k_nearest_neighbors(m, k):
    num_tensors = len(m)
    neighbors_list = []  # 存储每个张量的k近邻索引
    # 计算每个张量与其他张量之间的距离
    dists = torch.cdist(m, m, p=2)
    # 遍历每个张量
    for i in range(num_tensors):
        # 获取当前张量与其他张量的距离
        curr_dists = dists[i]
        # 排除自身距离
        curr_dists[i] = float('inf')
        # 找到最近的k个邻居索引
        similar_values, topk_indices = curr_dists.topk(k, largest=False)
        similarities = 1 / (1 + similar_values.float())
        value_list = similarities.tolist()
        index_list = topk_indices.tolist()
        for index, value in zip(index_list, value_list):
            neighbors_list.append([i] + [index] + [round(value, 5)])

    return neighbors_list


def find_k_nearest_neighbors_list(tensor, k=1, threshold=0.9):
    num_tensors = len(tensor)
    neighbors_list = []  # 存储每个张量的k近邻索引
    similar_list = []
    # 计算每个张量与其他张量之间的距离
    dists = torch.cdist(tensor, tensor, p=2)
    # 遍历每个张量
    for i in range(num_tensors):
        # 获取当前张量与其他张量的距离
        curr_dists = dists[i]
        # 排除自身距离
        curr_dists[i] = float('inf')
        # 找到最近的k个邻居索引
        similar_values, topk_indices = curr_dists.topk(k, largest=False)
        similarities = 1 / (1 + similar_values.float())
        value_list = similarities.tolist()
        index_list = topk_indices.tolist()
        for index, value in zip(index_list, value_list):
            if value >= 0.99:
                neighbors_list.append([i] + [index])
                if value == 1:
                    similar_list.append(0.99999)
                else:
                    similar_list.append(round(value, 5))
                # similar_list.append(2)

    print("add knn edge size:" + str(len(similar_list)))

    return neighbors_list, similar_list


def find_top_similar_rows(tensor, k=1, threshold=0.999):
    has_nan = torch.isnan(tensor).any()

    if has_nan or has_nan:
        tensor = torch.nan_to_num(tensor, nan=0.0)
        print("张量中有空值")
    else:
        print("张量中没有空值")
    neighbors_list = []  # 存储每个张量的k近邻索引
    similar_list = []
    # 计算相似度矩阵
    if not tensor.is_floating_point():
        tensor = tensor.float()
    similarity_matrix = cosine_similarity_matrix(tensor)

    # 创建一个掩码来排除自身的相似度
    mask = torch.eye(tensor.shape[0], device=tensor.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))

    # 找到每行的前k个最相似的行
    top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    high = top_k_similarities[top_k_similarities >= threshold].tolist()
    # while (len(high) > tensor.shape[0]):
    #     if threshold >= 0.99 :
    #         k = 1
    #         top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    #         break
    #     else:
    #         threshold = threshold + 0.01
    #         k = 1
    #         top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    #         high = top_k_similarities[top_k_similarities >= threshold].tolist()
    while True:
        top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        high = top_k_similarities[top_k_similarities >= threshold].tolist()
        if len(high) < similarity_matrix.shape[0]:
            break
        if threshold >= 0.99999:
            threshold = 0.99999
            break
        elif threshold >= 0.9999:
            threshold += 0.00001
        elif threshold >= 0.999:
            threshold += 0.0001
        elif threshold >= 0.99:
            k=1
            threshold += 0.001
        elif threshold >= 0.9:
            threshold += 0.01
        else:
            threshold += 0.1

    results = []
    for i in range(tensor.shape[0]):
        row_similarities = top_k_similarities[i]
        row_indices = top_k_indices[i]

        # 过滤掉相似度低于阈值的行
        valid_mask = row_similarities >= threshold
        valid_similarities = row_similarities[valid_mask].tolist()
        valid_indices = row_indices[valid_mask].tolist()

        if len(valid_indices) > 0:
            for index, value in zip(valid_indices, valid_similarities):
                if value >= threshold:
                    if not neighbors_list.__contains__([i] + [index]):
                        neighbors_list.append([i] + [index])
                        if value == 1:
                            value = 0.99999
                        else:
                            value = round(value, 5)
                        similar_list.append(value)
                        if not neighbors_list.__contains__([i] + [index]):
                            neighbors_list.append([index] + [i])
                            similar_list.append(value)

    print("add knn edge size:" + str(len(similar_list)))


    return neighbors_list, similar_list

# 计算余弦相似度
def cosine_similarity_matrix(tensor):
    # 标准化每一行
    normalized = F.normalize(tensor, p=2, dim=1)
    # 计算余弦相似度矩阵
    similarity = torch.mm(normalized, normalized.t())
    return similarity

def euclidean_distance_matrix(tensor):
    # 计算向量之间的欧氏距离矩阵
    dist_matrix = torch.cdist(tensor, tensor, p=2)
    return dist_matrix

def draw(graph):
    nids = graph.n_id
    graph = to_networkx(graph)
    for i, nid in enumerate(nids):
        graph.nodes[i]['txt'] = str(nid.item())
    node_labels = nx.get_node_attributes(graph, 'txt')
    # print(node_labels)
    # {0: '14', 1: '32', 2: '33', 3: '18', 4: '30', 5: '28', 6: '20'}
    nx.draw_networkx(graph, labels=node_labels, node_color='#00BFFF')
    plt.axis("off")
    plt.show()


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


if __name__ == '__main__':
    # # 示例用法
    # m = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0], [1.0, 2.0, 4.0], [3.0, 4.0, 5.0]])
    # k = 2
    #
    # neighbors, values = find_k_nearest_neighbors_list(m, k)
    # print(neighbors)
    # print(values)
    #
    # print(m.shape[0])

    # 假设我们有一个形状为 (1000, 2) 的张量 m
    # m = torch.randn(1000, 2)
    m = torch.tensor([[1.0,1],[1999,2678],[1,1],[1999,2678]])
    # 找到每行的前5个最相似的行，相似度阈值为90%
    neighbors_list, similar_list = find_top_similar_rows(m, k=1, threshold=0.9)
    # 打印结果
    print(neighbors_list)


