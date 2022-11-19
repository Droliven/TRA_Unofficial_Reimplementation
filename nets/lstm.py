#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      lstm.py
@time:      2022-11-14 14:45
@license:   Apache Licence
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):

    """LSTM Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        use_attn=True,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.noise_level = noise_level

        self.input_drop = nn.Dropout(input_drop)

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        '''

        :param x: [n, 60, 16]
        :return:
        '''
        x = self.input_drop(x)

        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level

        rnn_out, _ = self.rnn(x) # [n, 60, 64]
        last_out = rnn_out[:, -1]

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1) # [b, 60, 1] 注意这个 attention 就是这么来的
            att_out = (rnn_out * scores).sum(dim=1).squeeze() # [b, 64]
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out
