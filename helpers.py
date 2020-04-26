import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def normalize(train_input, test_input):
    mu, std = train_input.mean(), train_input.std()
    train_input_norm = train_input.sub_(mu).div_(std)
    test_input_norm = test_input.sub_(mu).div_(std)
    return train_input_norm, test_input_norm