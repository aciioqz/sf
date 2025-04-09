from torch_frame import stype
from torch_frame.typing import DataFrame
from torch_frame.utils import infer_series_stype

def infer_df_stype(df: DataFrame) :
    """Infer :obj:`col_to_stype` given :class:`DataFrame` object.

    Args:
        df (DataFrame): Input data frame.

    Returns:
        col_to_stype: Inferred :obj:`col_to_stype`, mapping a column name to
            its inferred :obj:`stype`.
    """
    col_to_stype = {}
    for col in df.columns:
        stype = infer_series_stype(df[col])
        if stype is not None:
            if stype is stype.multicategorical:
                col_to_stype[col] = stype.text_embedded
            else:
                col_to_stype[col] = stype


    return col_to_stype