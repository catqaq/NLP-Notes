#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:07:18 2019

@author: jjg
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import copy
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr


def training(regression, train_iter,dev_iter,model,device,lr,l2,epochs,static=False):
    #move model to device before constructing optimizer for it.
    model.to(device)
    if static:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=l2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    total_step = len(train_iter)
    train_accs = []
    dev_accs = []
    best_acc = float('inf') if regression else 0
    t0 = time()
    for epoch in range(1, epochs + 1):
        #training mode, we should reset it to training mode in each epoch
        model.train()
        for i, batch in enumerate(train_iter):
            texts, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(texts)

            if regression:
                loss = F.mse_loss(outputs, labels)  #regression
            else:
                loss = F.cross_entropy(outputs, labels)  #classification
            loss.backward()
            optimizer.step()

            #Visualization of the train process
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, epochs, i + 1, total_step, loss.item()))
        #evaluate the train set
        train_accs.append(evaluating(regression, train_iter, model, device))
        #in each epoch we call evaluating(), switch to evaluation mode
        dev_acc = evaluating(regression, dev_iter, model, device)
        dev_accs.append(dev_acc)
        # save the best model instead of the last epoch's model
        if regression:
            dev_acc < best_acc
            best_acc=dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        elif dev_acc > best_acc:
            best_acc = dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    t1 = time()
    print('training time: %.2f' % (t1 - t0))
    print('best result on dev_set: %.3f' % best_acc)
    show_training(regression, train_accs, dev_accs)
    return model


def evaluating(regression, data_iter, model, device):
    model.to(device)
    model.eval()  #evaluation mode
    size = len(data_iter.data_iter.dataset)
    with torch.no_grad():
        correct, avg_loss = 0, 0
        for batch in data_iter:
            #0代表x部分，1代表y部分
            texts, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(texts)
            if regression:
                loss = F.mse_loss(outputs, labels)  #regression
                avg_loss += loss.item()
            else:
                loss = F.cross_entropy(outputs, labels)  #classification
                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels).sum().item()

        accuracy = correct / size
        avg_loss /= size

        return avg_loss if regression else accuracy


def show_training(regression, train_accs, dev_accs):
    #plot train acc and validation acc
    plt.ion()
    plt.figure()
    metric = 'loss' if regression else 'acc'
    plt.ylabel(metric)
    plt.xlabel('epochs')
    plt.title('Training and validation' + metric)
    plt.tight_layout()
    #plt.xticks(range(0,args.epochs),range(1,args.epochs+1))
    plt.plot(train_accs, label='train_' + metric)
    plt.plot(dev_accs, label='dev_' + metric)
    plt.legend()


#对有特殊评价指标的任务，单独处理

def evaluate_mnli():
    #要分matched/mismatched
    pass


def evaluate_sts(targets, preds):
    #regression metric: Pearson and Spearman correlation
    corr = pearsonr(targets, preds)[0]
    print("pearson r: %.3f" % corr)
    corr = spearmanr(targets, preds)[0]
    print("spearman r: %.3f" % corr)


def evaluate_cola(targets, preds):
    #matthews_corrcoef
    mcc = matthews_corrcoef(targets, preds)
    print("mcc: %.3f" % mcc)


def f1(targets, preds):
    #对类别不平衡问题，除了acc之外，还需要f1
    f1 = f1_score(targets, preds)
    print("f1: %.3f" % f1)


class BatchWrapper(object):
    """对batch做个包装，方便调用"""

    def __init__(self, regression, data_iter, x_vars, y_vars, include_lengths=False):
        self.data_iter, self.x_vars, self.y_vars = data_iter, x_vars, y_vars
        self.include_lengths = include_lengths

    def __iter__(self):
        for batch in self.data_iter:
            if self.y_vars is not None:
                temp = [getattr(batch, feat) for feat in self.y_vars]
                if regression:
                    label = torch.cat(temp)
                else:
                    #classification, keep long int
                    #合并并不会改变dtype，加.long()只是为了排除label数据类型可能发生的改变
                    label = torch.cat(temp).long()
            else:
                raise ValueError('BatchWrapper: invalid label')
            #获取所有x属性列表，单句一个属性，双句两个属性
            fs = [getattr(batch, x_var) for x_var in self.x_vars]

            if self.include_lengths:
                #堆叠多个属性
                text = torch.stack([f[0] for f in fs])
                length = torch.stack([f[1] for f in fs])
                yield (text, label, length)
            else:
                text = torch.stack(fs)
                yield (text, label)

    def __len__(self):
        return len(self.data_iter)
