import torch
from torch import nn
from helpers import compute_stats, boxplot_accuracy
import CNN 
import WS 
import WS_AL 
import random

'''
Here, we train the models with the best set of parameters that we found using the grid search method (see tuning.py file) 
and display the average and standard deviations of each models over 15 rounds, as well as a boxplot to visualy see the improvement 
between the first and the third model. 
'''

# Setting the seed for reproducible results
torch.manual_seed(113)
torch.cuda.manual_seed(113)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Models to compare
models = [CNN.ConvNet, WS.WS_Net, WS_AL.AuxLoss_Net]

# Statistics for each model (average and std)
stats, labels_bp = compute_stats(models, nb_rounds=15, nb_samples=1000, nb_epochs=50, mini_batch_size=100, criterion=nn.CrossEntropyLoss(), eta=.5, eta2=1e-2, alpha=.65, gamma=.75, lossplot=False)

# Display performances comparison (boxplot)
boxplot_accuracy(stats, labels_bp)