import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self


class Table:
    r"""A table in a database.

    Args:
        df (pandas.DataFrame): The underyling data frame of the table.
        fkey_col_to_pkey_table (Dict[str, str]): A dictionary mapping
            foreign key names to table names that contain the foreign keys as
            primary keys.
        pkey_col (str, optional): The primary key column if it exists.
            (default: :obj:`None`)
        time_col (str, optional): The time column. (default: :obj:`None`)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fkey_col_to_pkey_table: Dict[str, str],
        pkey_col: Optional[str] = None,
        time_col: Optional[str] = None,
        is_need_edge: bool = True
    ):
        self.df = df
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.is_need_edge = is_need_edge

    def __repr__(self) -> str:
        return (
            f"Table(df=\n{self.df},\n"
            f"  fkey_col_to_pkey_table={self.fkey_col_to_pkey_table},\n"
            f"  pkey_col={self.pkey_col},\n"
            f"  time_col={self.time_col},\n"
            f"  is_need_edge={self.is_need_edge}"
            f")"
        )

    def __len__(self) -> int:
        r"""Returns the number of rows in the table."""
        return len(self.df)
