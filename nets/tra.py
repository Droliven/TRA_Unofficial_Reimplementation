#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      tra.py
@time:      2022-11-14 13:34
@license:   Apache Licence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TRA(nn.Module):
    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
    """

    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        if num_states > 1:
            self.router = nn.LSTM(input_size=num_states, hidden_size=hidden_size, num_layers=1, batch_first=True, )
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        self.predictors = nn.Linear(input_size, num_states)

    def forward(self, hidden, hist_loss):
        # [b, 512], [b, 60-21, 3]
        preds = self.predictors(hidden) # [b, 3]

        if self.num_states == 1:
            return preds.squeeze(-1), preds, None

        # information type
        router_out, _ = self.router(hist_loss) # [b, 60-21, 16]
        if "LR" in self.src_info:
            latent_representation = hidden # [b, 512]
        else:
            latent_representation = torch.randn(hidden.shape).to(hidden)
        if "TPE" in self.src_info:
            temporal_pred_error = router_out[:, -1] # [b, 16]
        else:
            temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)

        out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1)) # [b, 3], 把这个特征当做 gumbel 分布的概率
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

        if self.training:
            final_pred = (preds * prob).sum(dim=-1) # [b]
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)] # [] # todo: 测试的时候，为什么要这么取

        return final_pred, preds, prob
