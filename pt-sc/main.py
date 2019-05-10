#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:16:49 2019

@author: jjg
"""

from __future__ import unicode_literals, print_function, division
import torch
import random
from configs import device
from helper import prepareData
import models
from train_evaluate import trainIters, evaluateRandomly, evaluateInput


input_lang, output_lang, pairs = prepareData('pt', 'sc', True)  #普通话-四川话
print(random.choice(pairs))


if __name__ == '__main__':
    #hyper-parameters
    # #other parameters can be found in configs.py, but better to keep default
    use_model = False     #True:use the trained model, else training from scratch
    hidden_size = 256    #rnn's hidden_size
    in_embed_dim = 256   #input language word embedding dimension
    out_embed_dim = 256  #output language word embedding dimension
    lr = 0.01
    n_iters = 80000
    print_every = 1000
    plot_every = 100
    
    
    encoder1 = models.EncoderRNN(input_lang.n_words, hidden_size, in_embed_dim).to(device)
    attn_decoder1 = models.AttnDecoderRNN(hidden_size, output_lang.n_words, out_embed_dim, dropout_p=0.1).to(device)
    
    if use_model:
        encoder1.load_state_dict(torch.load('data/encoder_25.pt'))
        attn_decoder1.load_state_dict(torch.load('data/attn_decoder_25.pt'))
    else:
        trainIters(pairs, input_lang, output_lang, encoder1, attn_decoder1, n_iters=n_iters, print_every=print_every, plot_every=plot_every, learning_rate=lr)
    
    evaluateRandomly(pairs, input_lang, output_lang, encoder1, attn_decoder1)
    
    evaluateInput(input_lang, output_lang, encoder1, attn_decoder1)
    
    
    
    
    
    