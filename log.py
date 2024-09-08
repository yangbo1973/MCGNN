#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Date  : 2021/12/3
# @Name  : ZhouZongXin

"""
log格式封装

self.log_file_handler = TimedRotatingFileHandler(self.logNamePath, when="S", interval=1, backupCount=3)

when：是一个字符串，用于描述滚动周期的基本单位，字符串的值及意义如下：
“S”: Seconds
“M”: Minutes
“H”: Hours
“D”: Days
“W”: Week day (0=Monday)
“midnight”: Roll over at midnight
interval: 滚动周期，单位有when指定，比如：when=’D’,interval=1，表示每天产生一个日志文件；
backupCount: 表示日志文件的保留个数；
"""
import logging
import time
import os
from logging.handlers import TimedRotatingFileHandler

# BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# LOG_PATH = os.path.join(BASE_PATH, 'log')
import torch


class Logger(object):

    def __init__(self):
        self.logName = time.strftime('%Y-%m-%d %X')
        self.logNamePath = os.path.join('', '%s.log'%time.strftime('%Y_%m_%d_%H_%M_%S'))
        # 创建记录器
        self.logger = logging.getLogger("log")

        # debug阈值是10，设置低于10的将忽略过滤掉
        debug = logging.DEBUG
        self.logger.setLevel(debug)

        if not self.logger.handlers:
            # 这里设置成TimedRotatingFileHandler
            # 创建一个由时间控制的文件操作符：每天生成一个文件
            self.log_file_handler = TimedRotatingFileHandler(self.logNamePath, when="D", interval=1, backupCount=3)

            self.fileFormat = logging.Formatter("%(asctime)s-%(pathname)s [line:%(lineno)d]-> %(levelname)s: %(message)s")

            # 将格式化程序添加到控制台处理程序
            self.log_file_handler.setFormatter(self.fileFormat)

            # 添加到记录器
            self.logger.addHandler(self.log_file_handler)
            # torch.from_numpy()


logger = Logger().logger

if __name__ == '__main__':
    stop = {0 : "123"}
    logger.info("test start")
    logger.debug("test stop")
    logger.error('test: %s', stop)

