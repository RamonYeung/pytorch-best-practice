# Standard Modules
import random
import os
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
# 3rd-party Modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# self-defined Modules
from config import get_args
from model import LSTMClassifier
from dataset import MetaQA
from utils import set_seed, snapshot, show_params, l1_penalty
from metric import CategoricalAccuracy, PRMetric

args = get_args()

for name, arg in vars(args).items():
    print('%s: %s' % (name, arg))

# Reproducibility
set_seed(args.seed)
# 1. Define training device
train_device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
print(train_device)
# 2. Choose a dataset
dataset = MetaQA('1')
train_data = TensorDataset(dataset.train_x, dataset.train_l, dataset.train_y)
test_data = TensorDataset(dataset.test_x, dataset.test_l, dataset.test_y)
# 3. Choose a model
model = LSTMClassifier(train_device, len(dataset.vocab_q) + 1, args.embedding_size, args.hidden_size)
show_params(model)
# 4. Choose a criterion
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
# 5. Choose an optimizer
optimizer = optim.Adam(model.parameters(), args.lr)
# 6. Iterate dataset and begin training
train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=3)
test_loader = DataLoader(test_data, 2048, shuffle=False, num_workers=3)
# 7. Initialize metrics
train_acc = CategoricalAccuracy()
test_acc = CategoricalAccuracy()
pr = PRMetric(13)

for epoch in range(1, args.epochs + 1):
    model.train()
    model.to(train_device)
    # main loop
    for batch_i, (sent, length, label) in enumerate(train_loader):
        sent = sent.to(train_device)
        length = length.to(train_device)
        label = label.to(train_device)
        pred = model(sent, length)
        loss = criterion(pred, label)
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'INFO: [{epoch:02d}/{batch_i:04d}], avgloss: {loss.item():.4f}', end=', ')
        # training acc
        train_acc.update((pred, label))
        pr.update((pred, label))
        print(f'Training acc: {train_acc.compute() * 100:.2f}%')

    # testing acc
    model.eval()
    model.to(torch.device('cpu'))
    for batch_i, (sent, length, label) in enumerate(test_loader):
        with torch.no_grad():
            pred = model(sent, length)
            test_acc.update((pred, label))
    print(f'              Testing acc: {test_acc.compute() * 100:.2f}%')

    # saving models
    if test_acc.compute() > 0.9:
        snapshot(model, epoch, args.save_path)
p, r = pr.compute()
print(f'P: {p} \n R: {r}')