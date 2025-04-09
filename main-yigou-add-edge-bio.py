import copy
import os
# Stores the informative text columns to retain for each table:
import time
from typing import Dict

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, L1Loss, CrossEntropyLoss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from t4g.data import BenchDataset
from t4g.data.table import Table
from t4g.data.task import TaskType
from t4g.datasets import get_dataset
from t4g.encoder.text_embedder import GloveTextEmbedding
from t4g.heterogeneous_model.graph import get_stype_proposal, get_node_train_table_input, make_pkey_fkey_graph_add_edge
from t4g.heterogeneous_model.model import ModelW

import resource

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (60000, hard))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    def __init__(self, epochs=300, dataset=None, task=None, batch_size=128, num_layers=2):
        self.dataset = dataset
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.channels = 512
        self.aggr = "sum"
        self.num_neighbors = 128
        self.num_layers = num_layers
        self.max_steps_per_epoch = 2000
        self.num_workers = 1


def main(seed_value, args, root_dir, raw_file_root_path):
    start_timestamp = time.time()
    print("当前时间戳:", start_timestamp)
    seed_everything(seed_value)

    dataset_to_informative_text_cols = {}
    dataset_to_informative_text_cols[args.dataset] = {
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset: BenchDataset = get_dataset(name=args.dataset, process=True, path=raw_file_root_path)
    task = dataset.get_task(args.task)

    print("开始除了字段")
    col_to_stype_dict = get_stype_proposal(dataset.db)
    informative_text_cols: Dict = dataset_to_informative_text_cols[args.dataset]
    for table_name, stype_dict in col_to_stype_dict.items():
        for col_name, stype in list(stype_dict.items()):
            # 数据类型中删除label字段
            if table_name == task.entity_table and col_name == task.target_col:
                del stype_dict[col_name]

    # count_1 = np.sum(train_y)
    # count_0 = train_y.size - count_1
    # print(f"count_0: {count_0}, count_1: {count_1}")

    print("Create the graph ")

    data, col_stats_dict = make_pkey_fkey_graph_add_edge(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=device), batch_size=256),
        cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
    )

    cur_base_table: Table = copy.deepcopy(dataset.db.table_dict[task.entity_table])
    # split the base table to three part: train, val, test
    train_df, temp_df = train_test_split(cur_base_table.df, test_size=0.4)
    val_df, test_df = train_test_split(temp_df, test_size=0.5)

    loader_dict: Dict[str, NeighborLoader] = {}

    loader_dict["train"], train_table = get_loader_dict(data, cur_base_table, train_df, task, False, args)
    loader_dict["val"], val_table = get_loader_dict(data, cur_base_table, val_df, task, False, args)
    loader_dict["test"], test_table = get_loader_dict(data, cur_base_table, val_df, task, False, args)

    print("loader_dict")

    print("开始获取index")
    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        # weights = torch.tensor([count_0 / count_1], dtype=torch.float32).to(device)
        # loss_fn = BCEWithLogitsLoss(pos_weight=weights)
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        # Get the clamp value at inference time
        clamp_min, clamp_max = np.percentile(
            train_df[task.target_col].to_numpy(), [2, 98]
        )
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = 15
        loss_fn = CrossEntropyLoss()
        tune_metric = "accuracy"
        higher_is_better = True



    print("开始初始化Model")
    model = ModelW(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        norm="batch_norm",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    def train() -> float:
        model.train()

        loss_accum = count_accum = 0
        steps = 0
        total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
        current_position = 0
        for batch in tqdm(loader_dict["train"], total=total_steps):

            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            batch_size = args.batch_size

            end_position = current_position + batch_size
            if current_position + batch_size >= batch.y.size(0):
                pred = pred[:batch.y.size(0) - current_position]
                end_position = batch.y.size(0)
            else:
                pred = pred[:batch_size]

            label = batch.y[current_position:end_position].float()
            current_position += batch_size


            loss = loss_fn(pred.float(), label)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

            steps += 1
            if steps > args.max_steps_per_epoch:
                break

        return loss_accum / count_accum

    @torch.no_grad()
    def test(loader: NeighborLoader) -> np.ndarray:
        model.eval()

        pred_list = []
        current_position = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(
                batch,
                task.entity_table,
            )
            if task.task_type == TaskType.REGRESSION:
                assert clamp_min is not None
                assert clamp_max is not None


                pred = torch.clamp(pred, clamp_min, clamp_max)

            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)

            pred = pred.view(-1) if pred.size(1) == 1 else pred
            batch_size = args.batch_size
            end_position = current_position + batch_size
            if current_position + batch_size >= batch.y.size(0):
                pred = pred[:batch.y.size(0) - current_position]
                end_position = batch.y.size(0)
            else:
                pred = pred[:batch_size]
            current_position += batch_size
            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()

    loss_values = []
    acc_values = []
    state_dict = None
    best_val_metric = 0 if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        print(f"seed: {seed_value}, Epoch: {epoch:02d}")
        train_loss = train()
        val_pred = test(loader_dict["val"])
        val_metrics = task.heter_evaluate(val_pred, target_table=val_table)
        # print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

        if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] <= best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())

        loss_values.append(train_loss)
        # acc_values.append(val_metrics["roc_auc"])

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
        # torch.save(model, f'./model_seed_value_{seed_value}_epoch_{epoch}.pth')

    print(f"Best Val metrics: {val_metrics}")
    # plot_loss_with_acc(loss_values, acc_values)

    model.load_state_dict(state_dict)
    val_pred = test(loader_dict["val"])
    val_metrics = task.heter_evaluate(val_pred, target_table=val_table)
    print(f"Best Val metrics: {val_metrics}")

    test_pred = test(loader_dict["test"])
    test_metrics = task.heter_evaluate(test_pred, target_table=test_table)
    print(f"Best test metrics: {test_metrics}")

    current_timestamp = time.time()
    print("当前时间戳:", current_timestamp)

    peroid = current_timestamp - start_timestamp
    print("执行时间:", peroid)




def get_loader_dict(graph_data, base_table, table_df, task, shuffle, args):
    table: Table = Table(
        df=table_df,
        fkey_col_to_pkey_table=base_table.fkey_col_to_pkey_table,
        pkey_col=base_table.pkey_col
    )
    table_input = get_node_train_table_input(table, task)
    input_nodes = table_input.nodes
    transform = table_input.transform
    nei_loader = NeighborLoader(graph_data,
                                input_nodes=input_nodes,
                                transform=transform,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_neighbors=[int(args.num_neighbors / 2 ** i) for i in range(args.num_layers)],
                                num_workers=args.num_workers,
                                persistent_workers=args.num_workers > 0,
                                )
    return nei_loader, table


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


dataset =  "bio"
task = "bio-regress"
root_dir = "./data"
raw_file_root_path = os.path.join('./raw_data', 'bio', 'feature')
args = Args(batch_size = 64, epochs=200, dataset=dataset, task=task, num_layers=3)
seed_values = [42, 1234, 2043]
for seed in seed_values:
    print(seed)
    main(seed, args, root_dir, raw_file_root_path)





