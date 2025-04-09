from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import pandas as pd
from numpy.typing import NDArray

from t4g.data.database import Database
from t4g.data.table import Table

if TYPE_CHECKING:
    from t4g.data import Dataset



class Task:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        target_col: str,
        entity_table: str,
        entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
        entity_pk:List[str]
    ):
        self.dataset = dataset
        self.target_col = target_col
        self.metrics = metrics
        self.entity_table = entity_table
        self.entity_col = entity_col
        self.entity_pk = entity_pk

        self._full_test_table = None
        self._cached_table_dict = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"


    def evaluate(
        self,
        pred: NDArray,
        target: NDArray,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )
        return {fn.__name__: fn(target, pred) for fn in metrics}

    def heter_evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        return {fn.__name__: fn(target, pred) for fn in metrics}


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
    """
    REGRESSION = 'regression'
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multi_classification'
    MULTILABEL_CLASSIFICATION = 'multilabel_classification'

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION)


class BenchTask(Task):
    name: str
    task_type: TaskType
    entity_col: str
    entity_table: str
    time_col: str
    entity_pk:List[str]=[]

    timedelta: pd.Timedelta
    target_col: str
    metrics: List[Callable[[NDArray, NDArray], float]]

    task_dir: str = "tasks"

    task_path:str
    fkey_col_to_pkey_table:Dict

    def __init__(self, dataset, process: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            target_col=self.target_col,
            entity_table=self.entity_table,
            entity_col=self.entity_col,
            metrics=self.metrics,
            entity_pk = self.entity_pk
        )

