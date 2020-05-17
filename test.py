import torch
from torch import nn
from helpers import compute_stats, boxplot_accuracy
import CNN as CNN
import WS as WS
import WS_AL as WS_AL

models = [CNN.ConvNet3, WS.WS_Best_Net, WS_AL.AuxLossBest_Net]

stats, labels_bp = compute_stats(models, nb_rounds=15, nb_samples=1000, nb_epochs=50, mini_batch_size=100, criterion=nn.CrossEntropyLoss(), eta=9e-2, eta2=1e-4, alpha=.75, gamma=1, lossplot=False)

boxplot_accuracy(stats, labels_bp)