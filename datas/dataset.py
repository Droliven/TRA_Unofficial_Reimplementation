#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      dataset.py
@time:      2022-11-13 16:08
@license:   Apache Licence
"""
import copy
from typing import Callable, Union, List, Tuple, Text

import numpy as np
import pandas as pd
import torch

from datas.data_utils_func import lazy_sort_index, load_dataset, _to_tensor


def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    # assert index.is_lexsorted(), "index should be sorted"
    assert index.is_monotonic_increasing, "index should be sorted"

    # number of dates for each code # todo: 一共 2512790 条记录，为什么 CSI800 这里查出了 1670 只股票，最多的有 3141 天，最少的只有 10天
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0).size().values

    # start_index for each code
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # all the [start, stop) indices of features
    # features btw [start, stop) are used to predict the `stop - 1` label
    slices = []
    for cur_firstidx, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_firstidx + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices) # todo: 这些切片长短不统一

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn


class MemoryAugmentedTimeSeriesDataset():
    """Memory Augmented Time Series Dataset

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon (to mask historical loss for TRA)
        num_states (int): how many memory states to be added (for TRA)
        batch_size (int): batch size (<0 means daily batch)
        shuffle (bool): whether shuffle data
        pin_memory (bool): whether pin data to gpu memory
        drop_last (bool): whether drop last batch < batch_size
    """
    CS_ALL = "__all"  # return all columns with single-level index column
    CS_RAW = "__raw"  # return raw data with multi-level index column
    DK_I = "infer"  # the data processed for inference

    def __init__(
            self,
            longger,
            device,
            is_debug,
            handler,
            segments,
            seq_len=60,
            horizon=21,
            num_states=1,
            batch_size=512,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            **kwargs,
    ):

        assert horizon > 0, "please specify `horizon` to avoid data leakage"
        self.longger = longger
        self.device = device

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)  # for train/eval switch

        self.handler = handler.copy()
        self.segments = segments.copy()

        self.setup_data()


    def __repr__(self):
        return "{name}, segments={segments})".format(
            name=self.__class__.__name__, segments=self.segments
        )

    def setup_data(self):
        self.longger.info("Setup data ...")

        handler_data = pd.concat(
            {fields_group: load_dataset(path_or_obj) for fields_group, path_or_obj in self.handler["data_loader"]["config"].items()},
            axis=1,
            join="outer",
        ) # [2512790, 17]
        # change index to <code, date>
        handler_data.index = handler_data.index.swaplevel()
        handler_data = lazy_sort_index(handler_data, axis=0)

        self._data = handler_data["feature"].values.astype("float32") # [n, 16]
        self._label = handler_data["label"].squeeze().astype("float32") # [n]
        self._index = handler_data.index # [n]

        # add memory to feature
        self._data = np.c_[self._data, np.zeros((len(self._data), self.num_states), dtype=np.float32)] # [n, 16+numstage]

        # padding tensor
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32) # [60, 16+num_stage]

        # pin memory
        if self.pin_memory:
            self._data = _to_tensor(self._data, self.device)
            self._label = _to_tensor(self._label, self.device)
            self.zeros = _to_tensor(self.zeros, self.device)

        # ===== create batch slices =====
        self.batch_slices = _create_ts_slices(self._index, self.seq_len) # [2512790], 是不对齐的

        # create daily slices
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))} # 3141 天
        for i, (code, date) in enumerate(act_index):
            daily_slices[date].append(self.batch_slices[i])
        self.daily_slices = list(daily_slices.values())

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])

        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"This type of input is not supported")
        start_date = fn(start)
        end_date = fn(stop)
        # ===== 这个地方非常妙，根据参数返回一个可以枚举的新对象 =====
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data
        obj._label = self._label
        obj._index = self._index
        new_batch_slices = []
        for batch_slc in self.batch_slices:
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc)
        obj.batch_slices = np.array(new_batch_slices) # 1670400 / 324800 / 340790
        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices
        return obj

    def prepare(
            self,
            segments: Union[List[Text], Tuple[Text], Text, slice, pd.Index],
            col_set=CS_ALL,
            data_key=DK_I,
            **kwargs,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        Prepare the data for learning and inference.

        Parameters
        ----------
        segments : Union[List[Text], Tuple[Text], Text, slice]
            Describe the scope of the data to be prepared
            Here are some examples:

            - 'train'

            - ['train', 'valid']

        col_set : str
            The col_set will be passed to self.handler when fetching data.
            TODO: make it automatic:
                - select DK_I for test data
                - select DK_L for training data.
        data_key : str
            The data to fetch:  DK_*
            Default is DK_I, which indicate fetching data for **inference**.

        kwargs :
            The parameters that kwargs may contain:
                flt_col : str
                    It only exists in TSDatasetH, can be used to add a column of data(True or False) to filter data.
                    This parameter is only supported when it is an instance of TSDatasetH.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]:

        Raises
        ------
        NotImplementedError:
        """
        seg_kwargs = {"col_set": col_set}
        seg_kwargs.update(kwargs)
        self.longger.info(f"data_key[{data_key}] is ignored.")

        # Conflictions may happen here
        # - The fetched data and the segment key may both be string
        # To resolve the confliction
        # - The segment name will have higher priorities

        # 1) Use it as segment name first
        if isinstance(segments, str) and segments in self.segments:
            return self._prepare_seg(self.segments[segments], **seg_kwargs)

        if isinstance(segments, (list, tuple)) and all(seg in self.segments for seg in segments):
            return [self._prepare_seg(self.segments[seg], **seg_kwargs) for seg in segments]

        # 2) Use pass it directly to prepare a single seg
        return self._prepare_seg(segments, **seg_kwargs)

    def restore_index(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    def assign_data(self, index, vals):
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals, self.device)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -self.num_states:] = vals

    def clear_memory(self):
        self._data[:, -self.num_states:] = 0

    # TODO: better train/eval mode design
    def train(self):
        """enable traning mode"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""
        self.batch_size = -1
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:
            slices = self.daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:
            slices = self.batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # get slices for this batch
            slices_subset = slices[i: i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # collect data
            data = []
            label = []
            index = []
            for slc in slices_subset:
                _data = self._data[slc].clone() if self.pin_memory else self._data[slc].copy()
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                    else:
                        _data = np.concatenate([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                if self.num_states > 0:
                    _data[-self.horizon:, -self.num_states:] = 0
                data.append(_data)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)
            # concate
            index = torch.tensor(index, device=self.device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
            else:
                data = _to_tensor(np.stack(data), self.device)
                label = _to_tensor(np.stack(label), self.device)
            # yield -> generator
            # print(f"data: {data.shape}, label: {label.shape}, index: {index.shape}") # [b, 60, 17], [b], [b]
            yield {"data": data, "label": label, "index": index}

    # # helper functions
    # @staticmethod
    # def get_min_time(segments):
    #     return MemoryAugmentedTimeSeriesDataset._get_extrema(segments, 0, (lambda a, b: a > b))
    #
    # @staticmethod
    # def get_max_time(segments):
    #     return MemoryAugmentedTimeSeriesDataset._get_extrema(segments, 1, (lambda a, b: a < b))
    #
    # @staticmethod
    # def _get_extrema(segments, idx: int, cmp: Callable, key_func=pd.Timestamp):
    #     """it will act like sort and return the max value or None"""
    #     candidate = None
    #     for k, seg in segments.items():
    #         point = seg[idx]
    #         if point is None:
    #             # None indicates unbounded, return directly
    #             return None
    #         elif candidate is None or cmp(key_func(candidate), key_func(point)):
    #             candidate = point
    #     return candidate



if __name__ == '__main__':
    ds = MemoryAugmentedTimeSeriesDataset()
