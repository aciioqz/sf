from typing import Any, Dict, List

import pandas as pd
from torch_frame.utils import infer_series_stype
import random

def get_stype_proposal(df: pd.DataFrame, keys: List[str]) -> Dict[str, Any]:
    r"""Propose stype for columns of a table
    """
    inferred_col_to_stype_dict = {}
    none_type_columns = []
    for column in df.columns:
        if column not in keys:
            inferred_type = infer_series_stype(df[column])
            if inferred_type is None:
                none_type_columns.append(column)
            else:
                inferred_col_to_stype_dict[column] = inferred_type

    return inferred_col_to_stype_dict, none_type_columns


def split_and_save_csv(table_name: str, df: pd.DataFrame, exclude_columns: list):
    """
    Splits a CSV file into multiple smaller CSV files based on inferred column data types.

    :param file_path: Path to the input CSV file
    :param exclude_columns: List of column names to be excluded from the type inference
    :param output_folder: Folder where the output CSV files will be saved
    """
    column_types, none_type_columns = get_stype_proposal(df, exclude_columns)

    # Creating a dictionary to store DataFrames for each data type
    dataframes = {}

    result = {}
    # Separating columns by data type
    for column, dtype in column_types.items():
        if dtype not in dataframes:
            dataframes[dtype] = pd.DataFrame()
        dataframes[dtype][column] = df[column]

    for dtype, dataframe in dataframes.items():
        for column in exclude_columns:
            if column in df.columns:
                dataframe[column] = df[column]

    # Adding columns with None type to their own DataFrame
    if none_type_columns:
        if 'none' not in dataframes:
            dataframes['none'] = pd.DataFrame()
        for column in none_type_columns:
            dataframes['none'][column] = df[column]

    # Saving each DataFrame to a separate CSV file
    i = 0
    for dtype, dataframe in dataframes.items():
        temp_name = f"{table_name}_{i}"
        result[temp_name] = dataframe
        i = i + 1

    print("CSV files have been splited successfully.")
    return result


def split_dataframe(table_name: str, df: pd.DataFrame, keys: List[str]) -> Dict[str, pd.DataFrame]:
    """
    拆分DataFrame，每个新DataFrame中只包含主键、外键和一个普通特征，并保存为独立的CSV文件。

    参数:
    df (pd.DataFrame): 输入的DataFrame。
    keys (list): 主键和外键的列名数组。
    """

    # 获取列名
    columns = df.columns.tolist()

    # 获取普通特征的列名
    features = [col for col in columns if col not in keys]
    sub_tables = {}

    # 遍历每个普通特征列，创建新的DataFrame并保存为独立的CSV文件
    for index, feature in enumerate(features):
        if feature in keys:
            continue
        new_df = df[keys + [feature]]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
    return sub_tables
    print(f"split candidate table: {table_name} end！")

def split_dataframe_random(table_name: str, df: pd.DataFrame, keys: List[str]) -> Dict[str, pd.DataFrame]:
    """
    拆分DataFrame，每个新DataFrame中只包含主键、外键和一个普通特征，并保存为独立的CSV文件。

    参数:
    df (pd.DataFrame): 输入的DataFrame。
    keys (list): 主键和外键的列名数组。
    """

    # 获取列名
    columns = df.columns.tolist()

    # 获取普通特征的列名
    features = [col for col in columns if col not in keys]
    random.shuffle(features)
    sub_tables = {}
    groups = []

    while len(features) >= 2:
        # 随机选择两个特征
        group = random.sample(features, 2)
        groups.append(group)

        # 从原特征列表中移除所选特征
        for feature in group:
            features.remove(feature)
    if features:
        groups.append(features)


    for index, feature in enumerate(groups):
        if feature in keys:
            continue
        new_df = df[keys + feature]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
    return sub_tables
    print(f"split candidate table: {table_name} end！")

def split_dataframe_relation(table_name: str, df: pd.DataFrame, keys: List[str], relation_list:List[str]) \
        -> Dict[str, pd.DataFrame]:

    """
    df (pd.DataFrame): DataFrame。
    keys (list): [id, label]
    List[str]: ['Name-Pclass']
    """


    # 获取列名
    columns = df.columns.tolist()

    relation_list = [item.split('-') for item in relation_list]
    relation_set = set(item for sublist in relation_list for item in sublist)
    combined_set = relation_set.union(keys)

    # 获取普通特征的列名
    features = [col for col in columns if col not in combined_set]
    sub_tables = {}

    # 遍历每个普通特征列，创建新的DataFrame并保存为独立的CSV文件
    index = 0
    for feature in relation_list:
        new_df = df[keys + feature]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
        index = index + 1

    for feature in features:
        new_df = df[keys + [feature]]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
        index = index + 1
    return sub_tables
    print(f"split candidate table: {table_name} end！")

def add_foreign_key_columns(tables, id_column):
    all_fk_table = {}
    all_table_names = list(tables.keys())

    for target_table_name, target_df in tables.items():
        fk_table = {}
        for other_table_name in all_table_names:
            if other_table_name != target_table_name:
                other_ids = tables[other_table_name][id_column]
                # 为目标表增加一个新列，该列名为其他表的名称
                # target_df[other_table_name + '_id'] = other_ids.values
                target_df.loc[:, other_table_name + '_id'] = other_ids.values
                fk_table[other_table_name + '_id'] = other_table_name
        all_fk_table[target_table_name] = fk_table
        tables[target_table_name] = target_df

    return tables, all_fk_table


if __name__ == '__main__':

    # # 示例数据
    # table1 = pd.DataFrame({'id': [1, 2, 3], 'data1': ['A', 'B', 'C']})
    # table2 = pd.DataFrame({'id': [4, 5, 6], 'data2': ['D', 'E', 'F']})
    # table3 = pd.DataFrame({'id': [7, 8, 9], 'data3': ['G', 'H', 'I']})
    #
    # tables = {
    #     'table1': table1,
    #     'table2': table2,
    #     'table3': table3
    # }
    #
    # # 调用函数，增加外键
    # updated_tables, all_fk_table = add_foreign_key_columns(tables, 'id')
    #
    # # 输出结果
    # for table_name, df in updated_tables.items():
    #     print(f"Table: {table_name}")
    #     print(df)
    #     print()
    #
    # print(all_fk_table)

    import random

    # 原始的特征列表
    features = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']

    # 生成所有两两组合并随机打乱顺序
    random.shuffle(features)

    # 拆分成多组，每组包含两个特征
    groups = []

    while len(features) >= 2:
        # 随机选择两个特征
        group = random.sample(features, 2)
        groups.append(group)

        # 从原特征列表中移除所选特征
        for feature in group:
            features.remove(feature)

    # 如果最后剩下一个特征，将其单独放在最后一组
    if features:
        groups.append(features)

    # 遍历 groups 并输出每个组的索引和内容
    for index, group in enumerate(groups):
        print(f"Group {index}: {group}")
