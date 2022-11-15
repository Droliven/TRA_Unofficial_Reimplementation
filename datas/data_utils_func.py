#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      data_utils_func.py
@time:      2022-11-14 21:49
@license:   Apache Licence
"""
import os
import pandas as pd
from typing import Union, List
from packaging import version
import torch


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    make the df index sorted

    df.sort_index() will take a lot of time even when `df.is_lexsorted() == True`
    This function could avoid such case

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame:
        sorted dataframe
    """
    is_deprecated_lexsorted_pandas = version.parse(pd.__version__) > version.parse("1.3.0")

    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        or not is_deprecated_lexsorted_pandas
        and isinstance(idx, pd.MultiIndex)
        and not idx.is_lexsorted()
    ):  # this case is for the old version
        return df.sort_index(axis=axis)
    else:
        return df

def _to_tensor(x, device):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def get_level_index(df: pd.DataFrame, level=Union[str, int]) -> int:
    """

    get the level index of `df` given `level`

    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level

    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")

def load_dataset(path_or_obj, index_col=[0, 1]):
    """load dataset from multiple file formats"""
    if isinstance(path_or_obj, pd.DataFrame):
        return path_or_obj
    if not os.path.exists(path_or_obj):
        raise ValueError(f"file {path_or_obj} doesn't exist")
    _, extension = os.path.splitext(path_or_obj)
    if extension == ".h5":
        return pd.read_hdf(path_or_obj)
    elif extension == ".pkl":
        return pd.read_pickle(path_or_obj)
    elif extension == ".csv":
        return pd.read_csv(path_or_obj, parse_dates=True, index_col=index_col)
    raise ValueError(f"unsupported file type `{extension}`")

def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:
    """
    Time slicing in Qlib or Pandas is a frequently-used action.
    However, user often input all kinds of data format to represent time.
    This function will help user to convert these inputs into a uniform format which is friendly to time slicing.

    Parameters
    ----------
    t : Union[None, str, pd.Timestamp]
        original time

    Returns
    -------
    Union[None, pd.Timestamp]:
    """
    if t is None:
        # None represents unbounded in Qlib or Pandas(e.g. df.loc[slice(None, "20210303")]).
        return t
    else:
        return pd.Timestamp(t)


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    """
    fetch data from `data` with `selector` and `level`

    selector are assumed to be well processed.
    `fetch_df_by_index` is only responsible for get the right level

    Parameters
    ----------
    selector : Union[pd.Timestamp, slice, str, list]
        selector
    level : Union[int, str]
        the level to use the selector

    Returns
    -------
    Data of the given index.
    """
    # level = None -> use selector directly
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    # Try to get the right index
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[
                    pd.IndexSlice[idx_slc],
                ]
        else:  # pylint: disable=W0120
            return df
    else:
        return df.loc[
            pd.IndexSlice[idx_slc],
        ]


def fetch_df_by_col(df: pd.DataFrame, col_set: Union[str, List[str]]) -> pd.DataFrame:
    from datas.data_helper import DataHandler  # pylint: disable=C0415

    if not isinstance(df.columns, pd.MultiIndex) or col_set == DataHandler.CS_RAW:
        return df
    elif col_set == DataHandler.CS_ALL:
        return df.droplevel(axis=1, level=0)
    else:
        return df.loc(axis=1)[col_set]