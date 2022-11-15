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
import yaml

from nets.tra import TRA
from nets.lstm import LSTM
from datas.dataset import MemoryAugmentedTimeSeriesDataset

def evaluate(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label)
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
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
                 eval_test=False,
                 avg_params=True,
                 **kwargs
            ):

        self.logger = logger

        np.random.seed(seed)
        torch.manual_seed(seed)


        if model_type == "LSTM":
            self.model = LSTM(**model_config).to(device)

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

    def train_epoch(self, data_set):

        self.model.train()
        self.tra.train()

        data_set.train()

        max_steps = self.n_epochs
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, self.n_epochs)

        count = 0
        total_loss = 0
        total_count = 0
        for batch in tqdm(data_set, total=max_steps, desc=f"Train"):
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            data, label, index = batch["data"], batch["label"], batch["index"]
            # [b, 60, 17], [b], [b]
            feature = data[:, :, : -self.tra.num_states]
            hist_loss = data[:, : -data_set.horizon, -self.tra.num_states:]

            hidden = self.model(feature)  # [b, 512]
            pred, all_preds, prob = self.tra(hidden, hist_loss)  # [b], [b, 1], None

            loss = (pred - label).pow(2).mean()

            L = (all_preds.detach() - label[:, None]).pow(2)
            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            data_set.assign_data(index, L)  # save loss to memory

            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = self.lamb * (self.rho ** self.global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

        total_loss /= total_count

        return total_loss

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        metrics = []
        for batch in tqdm(data_set):  # [2088]
            data, label, index = batch["data"], batch["label"], batch["index"]
            # [800, 60, 17], [800], [800]
            feature = data[:, :, : -self.tra.num_states]
            hist_loss = data[:, : -data_set.horizon, -self.tra.num_states:]

            with torch.no_grad():
                hidden = self.model(feature)
                pred, all_preds, prob = self.tra(hidden, hist_loss)

            L = (all_preds - label[:, None]).pow(2)

            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            data_set.assign_data(index, L)  # save loss to memory

            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(),]
            columns = ["score", "label"]
            if prob is not None:
                X = np.c_[X, all_preds.cpu().numpy(), prob.cpu().numpy()]
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

    def fit(self, evals_result=dict()):

        train_set, valid_set, test_set = self.dataset.prepare(["train", "valid", "test"])
        # 1024 * [1632 / 318 / 333], [2512790, 17/1/1], [1670400], [324800], [340790]
        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {"model": copy.deepcopy(self.model.state_dict()), "tra": copy.deepcopy(self.tra.state_dict()), }
        params_list = {"model": collections.deque(maxlen=self.smooth_steps), "tra": collections.deque(maxlen=self.smooth_steps), }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.fitted = True
        self.global_step = -1
        # todo: 理清 Memory 机制
        if self.tra.num_states > 1:
            self.logger.info("init memory...")
            self.test_epoch(train_set)

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(train_set)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))
            self.tra.load_state_dict(average_params(params_list["tra"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if self.tra.num_states > 1 or self.eval_train:
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(train_set)[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("\ttrain metrics: %s" % train_metrics)

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {"model": copy.deepcopy(self.model.state_dict()), "tra": copy.deepcopy(self.tra.state_dict()), }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])
            self.tra.load_state_dict(params_list["tra"][-1])

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat({name: pd.DataFrame(evals_result[name]) for name in evals_result}, axis=1).to_csv(self.logdir + "/logs.csv", index=False)

            torch.save(best_params, self.logdir + "/model.bin")

            preds.to_pickle(self.logdir + "/pred.pkl")

            info = {
                "config": {"model_config": self.model_config, "tra_config": self.tra_config, "lr": self.lr, "n_epochs": self.n_epochs, "early_stop": self.early_stop, "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch, "lamb": self.lamb, "rho": self.rho, "seed": self.seed, "logdir": self.logdir, }, "best_eval_metric": -best_score,
                # NOTE: minux -1 for minimize
                "metric": metrics, }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = self.dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        return preds
