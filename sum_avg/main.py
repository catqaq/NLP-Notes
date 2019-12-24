#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:11:57 2019

@author: jjg
"""

from nltk import word_tokenize
import torch
from torchtext import data, datasets
from train_eval import training, evaluating, BatchWrapper
from configs import tasks, path_prefix
from models import LR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#choose task
task='SST'

if task not in tasks:
    print('error: please check the task name.')

data_path = path_prefix + task + '/'

TEXT = data.Field(
    sequential=True,
    tokenize=word_tokenize,
    lower=True,
    fix_length=35,   
    batch_first=True)

if task == 'SST':
    #SST,使用自带的，label是字符串,注意unk_token=None，否则label的词表头一个会是unk,导致label数字化后会从1开始，我们希望从0开始
    LABEL = data.Field(sequential=False, use_vocab=True, batch_first=True,unk_token=None)
elif task == 'STS-B':
    #回归问题，label一般为float，所以要指定dtype为float(默认为 torch.int64)
    # #原数据集中的score为float64, 但此处要指定为float(32)，因为outputs = model(texts)为float(32)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
else:
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True) #如果label为int的，则use_vocab=False

#single stencce or pair   
pair = True if tasks[task][0]==2 else False
fields=[('S1', TEXT), ('S2', TEXT), ('Label', LABEL)] if pair else [('text', TEXT), ('label', LABEL)]

x_fields = [f[0] for f in fields[:-1]]
y_fields = [f[0] for f in fields[-1:]]

#construct dataset
if task == 'SST':
    train, dev, test = datasets.SST.splits(TEXT, LABEL, root='data')
else:
    train, dev, test = data.TabularDataset.splits(
        path=data_path,
        train='train.tsv',
        validation='dev.tsv',
        test='test.tsv',
        format='tsv',
        fields=fields,
        skip_header=True)
    
#construct the vocab
TEXT.build_vocab(train, vectors="glove.840B.300d")
vocab = TEXT.vocab

#if label is str, we should build vocab too
if task=='SST':
    LABEL.build_vocab(train)

#construct data iter
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test),
    sort_key=lambda x: len(x.S1+x.S2) if tasks[task][0]==2 else len(x.text),  
    batch_sizes=(128, 128, 128),
    device=device)

train_iter = BatchWrapper(train_iter, x_fields, y_fields)
dev_iter = BatchWrapper(dev_iter, x_fields, y_fields)
test_iter = BatchWrapper(test_iter, x_fields, y_fields)

if __name__ == '__main__':
    regression = True if tasks[task][1] == 1 else False  #就1个回归问题，其实单独处理更好

    #hyper parameters
    learning_rate = 0.0001
    epochs = 50
    #fix_length = 50  依赖于任务
    #static = False  #update word emmbeddings or not,在模型和training中默认设置为static = False
    dropout = 0.5
    l2 = 0
    mean = True  #mean/sum

    t = 5
    test_accs = []
    for i in range(t):
        model = LR(task, vocab, mean, dropout=dropout)
        model = training(regression, train_iter, dev_iter, model, device,
                         learning_rate, l2, epochs)
        test_acc = evaluating(test_iter, model, device)[0]
        print('test_acc: %.3f' % test_acc)  #acc
        test_accs.append(test_acc)
    
    print('%d times: %s' % (t, test_accs))
    print('%d times average: %.3f' % (t, sum(test_accs) / t))
