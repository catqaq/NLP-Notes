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


def training(regression, train_iter, dev_iter, model, device, lr, l2, epochs, static=False):
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
    train_losses=[]
    dev_losses=[]
    best_acc = 0
    t0 = time()
    for epoch in range(1, epochs + 1):
        #training mode, we should reset it to training mode in each epoch
        model.train()
        for i, batch in enumerate(train_iter):
            texts, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(texts)

            if regression:
                loss = F.mse_loss(outputs, labels)  #回归
            else:
                loss = F.cross_entropy(outputs, labels)  #分类
            loss.backward()
            optimizer.step()

            #Visualization of the train process
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, epochs, i + 1, total_step, loss.item()))
        #evaluate the train set
        train_accs.append(evaluating(train_iter, model, device)[0])
        train_losses.append(evaluating(train_iter, model, device)[1])
        #in each epoch we call evaluating(), switch to evaluation mode
        dev_acc, dev_loss = evaluating(dev_iter, model, device)
        dev_accs.append(dev_acc)
        dev_losses.append(dev_loss)
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_model_wts = copy.deepcopy(
                model.state_dict())  #保存最优的model而非最后一轮的model

    model.load_state_dict(best_model_wts)
    t1 = time()
    print('training time: %.2f' % (t1 - t0))
    print('best acc on dev_set: %.3f' % best_acc)
    if regression:
        show_training(train_losses, dev_losses)
    else:
        show_training(train_accs, dev_accs)
    return model


def evaluating(data_iter, model, device):
    model.to(device)
    model.eval()  #evaluation mode
    with torch.no_grad():
        correct, avg_loss = 0, 0
        for batch in data_iter:
            texts, labels = batch[0].to(device), batch[1].to(
                device)  #0代表x部分，1代表y部分

            outputs = model(texts)
            predicted = torch.max(outputs.data, 1)[1]
            loss = F.cross_entropy(outputs, labels, reduction='mean')

            avg_loss += loss.item()
            correct += (predicted == labels).sum()

        size = len(data_iter.data_iter.dataset)
        avg_loss /= size
        accuracy = correct.item() / size
        #print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 100*accuracy, correct, size))
        return accuracy,avg_loss


def show_training(train_accs, dev_accs):
    #plot train acc and validation acc
    plt.ion()
    plt.figure()
    plt.ylabel('acc/loss')
    plt.xlabel('epochs')
    plt.title('Training and validation acc/loss')
    plt.tight_layout()
    #plt.xticks(range(0,args.epochs),range(1,args.epochs+1))
    plt.plot(train_accs, label='train_acc')
    plt.plot(dev_accs, label='dev_acc')
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

    def __init__(self, data_iter, x_vars, y_vars, include_lengths=False):
        self.data_iter, self.x_vars, self.y_vars = data_iter, x_vars, y_vars
        self.include_lengths = include_lengths

    def __iter__(self):
        for batch in self.data_iter:
            if self.y_vars is not None:
                temp = [getattr(batch, feat) for feat in self.y_vars]
                label = torch.cat(temp).long()
                #label = torch.cat(temp)  回归怎么改？
            else:
                raise ValueError('BatchWrapper: invalid label')

            fs = [getattr(batch, x_var).unsqueeze(1)
                  for x_var in self.x_vars]  #获取所有x属性列表，单句一个属性，双句两个属性

            if self.include_lengths:
                #堆叠多个属性
                text = torch.stack([f[0] for f in fs])  #保留长度时，TEXT字段为一个元组
                length = torch.stack([f[1] for f in fs])
                yield (text, label, length)
            else:
                text = torch.stack(fs)
                yield (text, label)

    def __len__(self):
        return len(self.data_iter)