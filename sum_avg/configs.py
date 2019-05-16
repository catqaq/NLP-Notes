#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:12:11 2019

@author: jjg
"""

#define data path
path_prefix = 'glue_data/'
names = [
    'CoLA', 'SST','SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'SNLI', 'QNLI', 'RTE',
    'WNLI'
]
details = [(1, 2), (1, 3),(1, 2), (2, 2), (2, 1), (2, 2), (2, 3), (2, 3), (2, 2),
           (2, 2), (2, 2)]

#统计各个任务的特点，最好都将标签数字化，方便处理，否则就需要记录每个任务的label是否为数字，以决定LABEL是否use_vocab
#tasks dict, key is task name, value is (single/pair, class)
#value元组第一个元素为1代表单句，为2代表句子对；第二个元素为1代表回归，为2代表二分类，为3代表3分类
tasks = dict(zip(names, details))