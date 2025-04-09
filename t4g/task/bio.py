import numpy as np
import pandas as pd
from tqdm import tqdm

from t4g.data import Database, BenchTask, Table
from t4g.data.task import TaskType
from t4g.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, macro_f1, micro_f1, mse, r2
from t4g.util.utils import get_df_in_window


class RegressTask(BenchTask):

    name = "bio-regress"
    task_type = TaskType.REGRESSION
    entity_col = "id"
    entity_table = "base"
    target_col = "logp"
    metrics = [mae, mse, rmse, r2]