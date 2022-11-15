#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      logging_help.py
@time:      2022-11-13 16:04
@license:   Apache Licence
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler # 按文件大小滚动备份
from logging import FileHandler # 按文件大小滚动备份
import colorlog


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class Log():
    def __init__(self,
                log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./log/',
                to_file=True):


        # 生成日志文件夹
        self.log_path = log_dir
        os.makedirs(self.log_path, exist_ok=True)

        # 生成日志名称
        now = str(datetime.now())
        file_name = now.split(" ")[0] + "_" + now.split(" ")[1][:8].replace(":", "-")
        self.logName = os.path.join(log_dir, log_file_name + '_' + file_name + '.log')  # 文件的命名
        # 生成formatter
        self.log_colors_config = {
            'DEBUG': 'green',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple',
        }
        # self.formatter = colorlog.ColoredFormatter('%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s', log_colors=self.log_colors_config)  # 日志输出格式
        self.color_formatter = colorlog.ColoredFormatter(' %(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s]- %(message)s ', log_colors=self.log_colors_config)  # 日志输出格式
        self.plain_formatter = logging.Formatter(' [%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s]- %(message)s ')  # 日志输出格式

        # 生成logger
        self.logger = logging.getLogger()
        self.logger.handlers.clear()

        # 创建一个StreamHandler,用于输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.color_formatter)
        self.logger.addHandler(stream_handler)

        if to_file:
            # 创建一个FileHandler，用于写到本地
            fh = FileHandler(filename=self.logName, mode='a', encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
            fh.setFormatter(self.plain_formatter)
            self.logger.addHandler(fh)

        self.logger.setLevel(log_level)


    def returnLogger(self):
        # 返回logger句柄
        return self.logger

if __name__ == '__main__':
    from tqdm import tqdm

    l = Log().returnLogger()
    # logger_init()
    for i in tqdm(range(100), total=100):
        if i < 20:
            l.debug(i)
        elif i < 40:
            l.info(i)
        elif i < 60:
            l.warning(i)
        elif i < 80:
            l.error(i)
        else:
            l.critical(i)

