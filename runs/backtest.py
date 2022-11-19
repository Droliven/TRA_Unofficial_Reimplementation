#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      backtest.py
@time:      2022-11-19 14:50
@license:   Apache Licence
"""
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import glob


def fmt(x, p=3, scale=1, std=False):
    '''

    :param x:
    :param p:
    :param scale:
    :param std:
    :return:
    '''
    _fmt = '{:.%df}'%p
    string = _fmt.format((x.mean() if not isinstance(x, (float, np.floating)) else x) * scale)
    if std and len(x) > 1:
        string += ' ('+_fmt.format(x.std()*scale)+')'
    return string


def backtest_func(x, N=80):
    '''
    返回 top 80 的收益率
    :param x: [800, 23]: [score, label, score_0-9, prob_0-9, ret]
    :param N:
    :return:
    '''
    ret = x.ret.copy()
    x = x.rank(pct=True, axis=0)
    x['ret'] = ret
    diff = x.score.sub(x.label)
    r = x.nlargest(N, columns='score').ret.mean()
    r -= x.nsmallest(N, columns='score').ret.mean()
    return pd.Series({
        'MSE': diff.pow(2).mean(),
        'MAE': diff.abs().mean(),
        'IC': x.score.corr(x.label),
        'R': r
    })


def backtest(pred_path=r"E:\PythonWorkspace\finc_tech\stock_ranking\tra_kdd21_reimplementation\ckpt\alstm_tra_init_1000\pred.pkl", ret_path=r"E:\PythonWorkspace\finc_tech\datas\ranking\tra_data\ret.pkl"):
    '''

    :return:
    '''
    # pred_path = os.path.join(logdir, "pred.pkl")
    pred = pd.read_pickle(pred_path).loc['2018-09-21':'2020-06-30']
    ret = pd.read_pickle(ret_path).clip(-0.1, 0.1)

    pred['ret'] = ret
    dates = pred.index.unique(level=0)
    res = Parallel(n_jobs=-1)(delayed(backtest_func)(pred.loc[d]) for d in dates)
    res = {
        dates[i]: res[i]
        for i in range(len(dates))
    }
    res = pd.DataFrame(res).T
    r = res['R'].copy()
    r.index = pd.to_datetime(r.index)
    r = r.reindex(pd.date_range(r.index[0], r.index[-1])).fillna(0)  # paper use 365 days
    output = {
               'MSE': res['MSE'].mean(),
               'MAE': res['MAE'].mean(),
               'IC': res['IC'].mean(),
               'ICIR': res['IC'].mean() / res['IC'].std(),
               'AR': r.mean() * 365,
               'AV': r.std() * 365 ** 0.5,
               'SR': r.mean() / r.std() * 365 ** 0.5,
               'MDD': (r.cumsum().cummax() - r.cumsum()).max()
           }
    output = pd.DataFrame([output])
    matric = {
        'MSE': fmt(output['MSE'], std=True),
        'MAE': fmt(output['MAE'], std=True),
        'IC': fmt(output['IC']),
        'ICIR': fmt(output['ICIR']),
        'AR': fmt(output['AR'], scale=100, p=1)+'%',
        'VR': fmt(output['AV'], scale=100, p=1)+'%',
        'SR': fmt(output['SR']),
        'MDD': fmt(output['MDD'], scale=100, p=1)+'%'
    }
    return matric, r

def backtest_multi(fname):
    res = []
    pnl = []
    metric, r = backtest(fname)
    res.append(metric)
    pnl.append(r)
    res = pd.DataFrame(res)
    pnl = pd.concat(pnl, axis=1)
    return {
        'MSE': fmt(res['MSE'], std=True),
        'MAE': fmt(res['MAE'], std=True),
        'IC': fmt(res['IC']),
        'ICIR': fmt(res['ICIR']),
        'AR': fmt(res['AR'], scale=100, p=1)+'%',
        'VR': fmt(res['AV'], scale=100, p=1)+'%',
        'SR': fmt(res['SR']),
        'MDD': fmt(res['MDD'], scale=100, p=1)+'%'
    }, pnl

if __name__ == '__main__':
    # exps = {
    #     # 'Linear': ['output/Linear/pred.pkl'],
    #     # 'LightGBM': ['output/GBDT/lr0.05_leaves128/pred.pkl'],
    #     # 'MLP': glob.glob('output/search/MLP/hs128_bs512_do0.3_lr0.001_seed*/pred.pkl'),
    #     # 'SFM': glob.glob('output/search/SFM/hs32_bs512_do0.5_lr0.001_seed*/pred.pkl'),
    #     # 'ALSTM': glob.glob('output/search/LSTM_Attn/hs256_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    #     # 'Trans.': glob.glob('output/search/Transformer/head4_hs64_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    #     # 'ALSTM+TS': glob.glob('output/LSTM_Attn_TS/hs256_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    #     # 'Trans.+TS': glob.glob('output/Transformer_TS/head4_hs64_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    #     'ALSTM+TRA(Ours)': r'E:\PythonWorkspace\finc_tech\stock_ranking\tra_kdd21_reimplementation\ckpt\alstm_tra_init_1000\pred.pkl',
    #     # 'Trans.+TRA(Ours)': glob.glob('output/search/finetune/Transformer_tra/K3_traHs16_traSrcLR_TPE_traLamb1.0_head4_hs64_bs512_do0.1_lr0.0005_seed*/pred.pkl')
    # }
    #
    # res = {
    #     name: backtest_multi(exps[name])
    #     for name in exps
    # }
    # report = pd.DataFrame({
    #     k: v[0]
    #     for k, v in res.items()
    # }).T

    a, b = backtest()
    pass


