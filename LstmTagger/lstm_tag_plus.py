#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 22:48:16 2019

@author: jjg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET",
                                                       "NN"])]

#construct the word_vocab, char_vocab, tag_vocab
word2ix = {}
char2ix = {}
tag2ix = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word2ix:
            word2ix[word] = len(word2ix)

        for char in word:
            if char not in char2ix:
                char2ix[char] = len(char2ix)
    for tag in tags:
        if tag not in tag2ix:
            tag2ix[tag] = len(tag2ix)

ix2tag = {v: k for k, v in tag2ix.items()}
#see the vocab's detail
print(word2ix)
print(char2ix)
print(tag2ix)
print(ix2tag)


class LSTMTaggerPlus(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, word_hidden_dim,
                 char_hidden_dim, word_vocab_size, char_vocab_size,
                 tagset_size):

        super(LSTMTaggerPlus, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size,
                                            word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size,
                                            char_embedding_dim)

        self.char_lstm = nn.LSTM(
            char_embedding_dim,
            char_hidden_dim)  #output char-level representation of each word.
        self.word_lstm = nn.LSTM(word_embedding_dim + char_hidden_dim,
                                 word_hidden_dim)  #concatenation

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.word_hidden = self.init_hidden(self.word_hidden_dim)
        self.char_hidden = self.init_hidden(self.char_hidden_dim)

    def init_hidden(self, size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # NOTE: LSTM's hidden has 2 term: (h_n, c_n)
        return (torch.zeros(1, 1, size), torch.zeros(1, 1, size))

    def forward(self, word_sequence, char_sequence):
        word_embeds = self.word_embeddings(word_sequence)

        char_embeds = self.char_embeddings(char_sequence)
        char_lstm_out, self.char_hidden = self.char_lstm(
            char_embeds.view(len(char_sequence), 1, -1), self.char_hidden)
        #char_lstm_out[-1],choose the last time step's output(through the full word)
        concat = torch.cat(
            [word_embeds.view(1, 1, -1), char_lstm_out[-1].view(1, 1, -1)],
            dim=2)
        word_lstm_out, self.word_hidden = self.word_lstm(
            concat, self.word_hidden)

        tag_space = self.hidden2tag(word_lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()


def test(model, test_example):
    with torch.no_grad():
        for word in test_example:
            word_sequence = prepare_sequence([word], word2ix)
            #print(word_sequence) single word's word_sequence, only single element
            char_sequence = prepare_sequence(word, char2ix)
            #print(char_sequence) #1*len(word)
            tag_scores = model(word_sequence, char_sequence)
            #print(tag_scores, tag_scores.argmax(dim=1).item())
            print('{}: {}'.format(word,
                                  ix2tag[tag_scores.argmax(dim=1).item()]))


def train(model, epochs):
    t0 = time()
    for epoch in range(epochs):
        for sentence, tags in training_data:
            # Step 1. clear the gradients.
            model.zero_grad()
            # Step 2. clear the gradients.
            # word hidden init by sentence
            model.word_hidden = model.init_hidden(model.word_hidden_dim)

            # Step 3. Run our forward pass on each word
            for index, word in enumerate(sentence):
                # NOTE: char hidden init by word
                # Clear hidden state between EACH word (char-level representation must be independent of previous word)
                model.char_hidden = model.init_hidden(model.char_hidden_dim)

                word_sequence = prepare_sequence([word], word2ix)
                char_sequence = prepare_sequence(word, char2ix)
                targets = prepare_sequence([tags[index]], tag2ix)

                tag_scores = model(word_sequence, char_sequence)

                loss = loss_function(tag_scores, targets)
                #NOTE: must accumulate gradients here, because we update parameters by sentence
                loss.backward(retain_graph=True)

            # Step 4. Update parameters at the end of sentence
            optimizer.step()
    print("%s epochs in %.2f sec for model at word level" % (epochs,
                                                             time() - t0))


if __name__ == '__main__':
    # These will usually be more like 32 or 64 dimensional.
    # we use small values to see the difference before-after the train
    WORD_EMBEDDING_DIM = 6
    WORD_HIDDEN_DIM = 6
    CHAR_EMBEDDING_DIM = 3
    CHAR_HIDDEN_DIM = 3
    learning_rate = 0.1
    epochs = 300

    #construct the model, loss function, optimizer
    model = LSTMTaggerPlus(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM,
                           CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(word2ix),
                           len(char2ix), len(tag2ix))

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    test(model, training_data[0][0])  #before train
    train(model, epochs)
    test(model, training_data[0][0])  #after train, correctly tagging