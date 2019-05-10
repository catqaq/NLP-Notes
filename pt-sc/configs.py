#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:55:43 2019

@author: jjg
"""
import torch
#settings

#any better way to set global variables that can be used across modules?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 15
teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1