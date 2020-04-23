import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def standardize(tensor):
    mu, std = tensor.mean(), tensor.std()
    return tensor.sub_(mu).div_(std)