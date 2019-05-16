#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:10:05 2019

@author: jjg
"""
import torch
import torch.nn as nn
from configs import tasks

class LR(nn.Module):
    def __init__(self,
                 task,
                 vocab,
                 mean,
                 static=False,
                 embed_dim=300,
                 dropout=0.5):
        
        super(LR, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        if static:
            self.embed.weight.requires_grad = False
        else:
            self.embed.weight.requires_grad = True

        self.mean = mean
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tasks[task][0] * embed_dim, tasks[task][1])

    def forward(self, x):
        if len(x) == 1:
            x = self.embed(x)
            #sum/avg
            x = x.squeeze(0)
            if self.mean:
                x = torch.mean(x, dim=2)
            else:
                x = torch.sum(x, dim=2)
            x = x.squeeze(1)
            
        elif len(x) == 2:
            if self.mean:
                x=torch.cat([torch.mean(self.embed(s), dim=2).squeeze() for s in x],dim=1)
            else:
                x=torch.cat([torch.sum(self.embed(s), dim=2).squeeze() for s in x],dim=1)
    
        x = self.dropout(x)
        x = self.fc(x)

        return x