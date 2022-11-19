#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      run.py
@time:      2022-11-14 13:39
@license:   Apache Licence
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import collections
import json
import pandas as pd
import copy
from joblib import Parallel, delayed
from tensorboardX import SummaryWriter

from nets.tra import TRA
from nets.lstm import LSTM
from nets.transformer import Transformer
from datas.dataset import MemoryAugmentedTimeSeriesDataset
from runs.backtest import backtest


def evaluate(pred):
    '''

    :param pred: # [b, 1+1 + 3 + 3]
    :return:
    '''
    pred = pred.rank(pct=True) # 根据 score 排序 # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label)
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    '''
    todo: 这个函数在干嘛
    :param params_list:
    :return:
    '''
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    '''
    一种最优传输理论的解法，目标是使得分布接近
    :param Q:
    :param n_iters:
    :param epsilon:
    :return:
    '''
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


class Run_TRAModel():
    def __init__(self,
                 logger,
                 device,
                 is_debug,
                 dataset_cfg,
                 model_config,
                 tra_config,
                 model_type="LSTM",
                 lr=2e-4,
                 n_epochs=500,
                 early_stop=20,
                 smooth_steps=5,
                 max_steps_per_epoch=100,
                 freeze_model=False,
                 model_init_state=None,
                 lamb=1.0,
                 rho=0.99,
                 seed=1000,
                 logdir=None,
                 eval_train=True,
                 eval_test=True,
                 avg_params=True,
                 **kwargs
            ):

        self.logger = logger

        if model_type == "LSTM":
            self.model = LSTM(**model_config).to(device)
        elif model_type == "Transformer":
            self.model = Transformer(**model_config).to(device)

        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info(f"Basic {model_type} params: {sum([p.numel() for p in self.model.parameters()]) / 1e6:>.8f} M")

        self.tra = TRA(self.model.output_size, **tra_config).to(device)
        self.logger.info(f"Tra plugin params: {sum([p.numel() for p in self.tra.parameters()]) / 1e6:>.8f} M")

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=lr)

        # 导入数据集
        self.dataset = MemoryAugmentedTimeSeriesDataset(longger=self.logger, device=device, **dataset_cfg)

        self.ret_path = dataset_cfg["handler"]["data_loader"]["config"]["ret"]
        self.model_config = model_config
        self.tra_config = tra_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        self.rho = rho
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params

        if self.tra.num_states > 1 and not self.eval_train:
            self.logger.warn("`eval_train` will be ignored when using TRA")

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.fitted = False
        self.global_step = -1

        self.summary_writer = SummaryWriter(logdir=self.logdir)


    def train_epoch(self, data_set):

        self.model.train()
        self.tra.train()

        data_set.train()

        max_steps = self.n_epochs
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, self.n_epochs)

        total_loss = 0
        total_mse_loss = 0
        total_assign_loss = 0
        sample_count = 0
        step_count = 0

        for batch in tqdm(data_set, total=max_steps, desc=f"Train"):
            step_count += 1
            if step_count > max_steps:
                break

            self.global_step += 1

            data, label, index = batch["data"], batch["label"], batch["index"]
            # [b, 60, 17], [b], [b]
            feature = data[:, :, : -self.tra.num_states] # [b, 60, 16]
            hist_loss = data[:, :-data_set.horizon, -self.tra.num_states:]

            hidden = self.model(feature)  # [b, 512]
            pred, all_preds, prob = self.tra(hidden, hist_loss)  # [b], [b, numstates], [b, numstates]

            mse_loss = (pred - label).pow(2).mean()

            L = (all_preds.detach() - label[:, None]).pow(2)
            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            data_set.update_memory(index, L)  # save loss to memory

            assign_loss = 0
            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = self.lamb * (self.rho ** self.global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                assign_loss = - lamb * reg

            loss = mse_loss + assign_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_assign_loss += assign_loss.item()
            sample_count += len(pred)

        total_loss /= sample_count
        total_mse_loss /= sample_count
        total_assign_loss /= sample_count

        return total_loss, total_mse_loss, total_assign_loss

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        metrics = []
        for batch in tqdm(data_set, desc=f"Test"):  # [2088]
            data, label, index = batch["data"], batch["label"], batch["index"]
            # [800, 60, 16+nums], [800], [800]
            feature = data[:, :, :-self.tra.num_states]
            hist_loss = data[:, :-data_set.horizon, -self.tra.num_states:] # [b, 39, 3], todo: 60-21 = 39

            with torch.no_grad():
                hidden = self.model(feature) # [b, 512]
                pred, all_preds, prob = self.tra(hidden, hist_loss) # [b]

            L = (all_preds - label[:, None]).pow(2) # [n, 3]

            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            data_set.update_memory(index, L)  # save loss to memory

            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(),] # [n, 1+1]
            columns = ["score", "label"]
            if prob is not None:
                X = np.c_[X, all_preds.cpu().numpy(), prob.cpu().numpy()] # [n, 1+1 + 3 + 3]
                columns += ["score_%d" % d for d in range(all_preds.shape[1])] + ["prob_%d" % d for d in range(all_preds.shape[1])]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)
            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {"MSE": metrics.MSE.mean(), "MAE": metrics.MAE.mean(), "IC": metrics.IC.mean(), "ICIR": metrics.IC.mean() / metrics.IC.std(), }

        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)

        return metrics, preds

    def fit(self):

        train_set, valid_set, test_set = self.dataset.prepare(["train", "valid", "test"])
        # 1024 * [1632 / 318 / 333], [2512790, 17/1/1], [1670400], [324800], [340790]
        best_score_ic = -1e20
        params_list = {"basic": collections.deque(maxlen=self.smooth_steps), "tra": collections.deque(maxlen=self.smooth_steps), } # 一直追加

        # train
        self.fitted = True
        self.global_step = -1
        # todo: 理清 Memory 机制
        if self.tra.num_states > 1:
            self.logger.info("init memory by testing on train dataset ...")
            init_metrics, init_preds = self.test_epoch(train_set)

        for epoch in range(self.n_epochs):
            # 1. 训练
            total_loss, total_mse_loss, total_assign_loss = self.train_epoch(train_set)

            self.summary_writer.add_scalar("Training/loss", total_loss, epoch)
            self.summary_writer.add_scalar("Training/mse_loss", total_mse_loss, epoch)
            self.summary_writer.add_scalar("Training/assign_loss", total_assign_loss, epoch)
            self.logger.info(f"Epoch {epoch}: Training | all loss {total_loss:>.6f} | mse loss {total_mse_loss:>.6f} | assign loss {total_assign_loss:>.6f}")

            # 2. 用滑动平均的参数验证 average params for inference
            params_list["basic"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            self.model.load_state_dict(average_params(params_list["basic"])) # todo: 这一步的作用是什么
            self.tra.load_state_dict(average_params(params_list["tra"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if self.tra.num_states > 1 or self.eval_train:
                # todo: 这里为什么要清空存储
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(train_set)[0]
                self.summary_writer.add_scalar("TestingTrain/MSE", train_metrics['MSE'], epoch)
                self.summary_writer.add_scalar("TestingTrain/MAE", train_metrics['MAE'], epoch)
                self.summary_writer.add_scalar("TestingTrain/IC", train_metrics['IC'], epoch)
                self.summary_writer.add_scalar("TestingTrain/ICIR", train_metrics['ICIR'], epoch)
                self.logger.info(f"Epoch {epoch}: Testing trainSet | MSE {train_metrics['MSE']:>.6f} | MAE {train_metrics['MAE']:>.6f} | IC {train_metrics['IC']:>.6f} | ICIR {train_metrics['ICIR']:>.6f}")

            valid_metrics = self.test_epoch(valid_set)[0]
            self.summary_writer.add_scalar("TestingVal/MSE", valid_metrics['MSE'], epoch)
            self.summary_writer.add_scalar("TestingVal/MAE", valid_metrics['MAE'], epoch)
            self.summary_writer.add_scalar("TestingVal/IC", valid_metrics['IC'], epoch)
            self.summary_writer.add_scalar("TestingVal/ICIR", valid_metrics['ICIR'], epoch)
            self.logger.info(f"Epoch {epoch}: Testing valSet | MSE {valid_metrics['MSE']:>.6f} | MAE {valid_metrics['MAE']:>.6f} | IC {valid_metrics['IC']:>.6f} | ICIR {valid_metrics['ICIR']:>.6f}")

            # restore parameters
            self.model.load_state_dict(params_list["basic"][-1])
            self.tra.load_state_dict(params_list["tra"][-1])
            # 在测试集 保存并回测模型
            test_metrics, test_preds = self.test_epoch(test_set, return_pred=True)
            self.summary_writer.add_scalar("TestingTest/MSE", test_metrics['MSE'], epoch)
            self.summary_writer.add_scalar("TestingTest/MAE", test_metrics['MAE'], epoch)
            self.summary_writer.add_scalar("TestingTest/IC", test_metrics['IC'], epoch)
            self.summary_writer.add_scalar("TestingTest/ICIR", test_metrics['ICIR'], epoch)
            self.logger.info(f"Epoch {epoch}: Testing testSet | MSE {test_metrics['MSE']:>.6f} | MAE {test_metrics['MAE']:>.6f} | IC {test_metrics['IC']:>.6f} | ICIR {test_metrics['ICIR']:>.6f}")

            # 保存并回测模型
            if test_metrics["IC"] > best_score_ic:
                best_score_ic = test_metrics["IC"]

                test_preds.to_pickle(os.path.join(self.logdir, f"pred.pkl"))
                torch.save({"basic": copy.deepcopy(self.model.state_dict()), "tra": copy.deepcopy(self.tra.state_dict())}, os.path.join(self.logdir, "model.bin"))

                self.logger.info(f"Epoch {epoch}: Best model and predictions SAVED and scores could be found ABOVE!")

                # 回测
                backtest_matric, pnl = backtest(pred_path=os.path.join(self.logdir, "pred.pkl"), ret_path=self.ret_path)
                self.summary_writer.add_scalar("Backtesting/MSE", float(backtest_matric['MSE']), epoch)
                self.summary_writer.add_scalar("Backtesting/MAE", float(backtest_matric['MAE']), epoch)
                self.summary_writer.add_scalar("Backtesting/IC", float(backtest_matric['IC']), epoch)
                self.summary_writer.add_scalar("Backtesting/ICIR", float(backtest_matric['ICIR']), epoch)
                self.summary_writer.add_scalar("Backtesting/AR%", float(backtest_matric['AR'][:-1]), epoch)
                self.summary_writer.add_scalar("Backtesting/VR%", float(backtest_matric['VR'][:-1]), epoch)
                self.summary_writer.add_scalar("Backtesting/SR", float(backtest_matric['SR']), epoch)
                self.summary_writer.add_scalar("Backtesting/MDD%", float(backtest_matric['MDD'][:-1]), epoch)
                self.logger.info(f"Epoch {epoch}: Backtesting ||- MSE {backtest_matric['MSE']} | MAE {backtest_matric['MAE']} | IC {backtest_matric['IC']} | ICIR {backtest_matric['ICIR']} -||- AR {backtest_matric['AR']} | VR {backtest_matric['VR']} | SR {backtest_matric['SR']} | MDD {backtest_matric['MDD']} -||")







