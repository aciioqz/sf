import json
import networkx as nx
import pandas
import os
import pandas as pd
import re
from networkx.algorithms.community import girvan_newman

def read_and_parse_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def normalize_values(data_dict):
    values = list(data_dict.values())
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        normalized_dict = {key: 1.0 for key, value in data_dict.items()}
    else:
        normalized_dict = {
            key: (value - min_value) / (max_value - min_value) if max_value != min_value else 0.0
            for key, value in data_dict.items()
        }
    return normalized_dict


def filter_by_threshold(normalized_dict, k):
    """过滤value值大于阈值k的key-value对"""
    filtered_dict = {
        key: value for key, value in normalized_dict.items() if value > k
    }
    return filtered_dict


import networkx as nx
from networkx.algorithms.community import girvan_newman


def graph_con_girvan_newman(column_names, keys):
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(column_names)

    # 添加边到图中
    for key in keys:
        node1, node2 = key.split('&&')
        if node1 in column_names and node2 in column_names:
            G.add_edge(node1, node2)
        else:
            print(f"Error: One or both of nodes '{node1}' and '{node2}' not in column_names")

    # 使用 Girvan-Newman 算法进行社区检测
    comp = girvan_newman(G)

    # 这里，我们只取第一层得到的社区划分
    communities = next(comp)
    community_lists = [list(c) for c in communities]
    return community_lists



def graph_con(column_names, keys):
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(column_names)

    # 添加边到图中
    for key in keys:
        node1, node2 = key.split('&&')
        if node1 in column_names and node2 in column_names:
            G.add_edge(node1, node2)
        else:
            print(f"Error: One or both of nodes '{node1}' and '{node2}' not in column_names")

    # 找到所有的最大团
    maximal_cliques = list(nx.find_cliques(G))

    return maximal_cliques


def split_df_by_column_list(df, column_lists, base_name, key_list, dest_path, k):

    # 创建一个字典用于存储列名到表名的映射
    column_to_tables = {}

    # 遍历每个列名列表
    for i, columns in enumerate(column_lists):
        # 创建一个子数据框，包括当前列列表和 'id' 列
        sub_df = df[key_list + columns]

        # 生成子数据框的文件名
        sub_csv_name = f"{base_name}_{i}.csv"

        # 保存子数据框为新的 CSV 文件
        sub_df.to_csv(os.path.join(dest_path, sub_csv_name), index=False)

        # 更新 column_to_tables 映射
        for col in columns:
            if col not in column_to_tables:
                column_to_tables[col] = []
            column_to_tables[col].append(sub_csv_name)

    # # 找到具有相同列的表并记录到文本中
    file_dict = {}
    for col, tables in column_to_tables.items():
        if len(tables) > 1:
            file_dict[col] = tables


    # 从txt文件中读取内容并解析
    if len(file_dict) > 0:
        file_path = f"{base_name}_same_columns_{k}.json"
        with open(file_path, 'w') as file:
            json.dump(file_dict, file, indent=4)
            print('end')

    if len(file_dict) > 0:
       for key, tables in file_dict.items():
           for table in tables:
               csv_path = os.path.join(dest_path, table)

               # 读取CSV文件
               df = pd.read_csv(csv_path)

               # 如果列不存在则添加列
               if key in df.columns:
                   fk_name = f'pk_{key}'
                   df[fk_name] = df[key_list[0]]
               df.to_csv(csv_path, index=False)


def split_handle(filename, k, csv_df, ids, dest_path, base_name, is_split:False):
    data_dict = read_and_parse_file(filename)
    if len(data_dict) ==0:
        maximal_cliques=[csv_df.columns.tolist()]
    else:
        normalized_dict = normalize_values(data_dict)
        filtered_dict = filter_by_threshold(normalized_dict, k)
        if len(filtered_dict) == 0:
            return []
        keys = list(filtered_dict.keys())
        column_names = csv_df.drop(columns=ids).columns.tolist()
        maximal_cliques = graph_con_girvan_newman(column_names, keys)
    print(maximal_cliques)
    if is_split:
        split_df_by_column_list(csv_df, maximal_cliques, base_name, ids, dest_path, k)
    return maximal_cliques