import torch
from torch import nn
from helpers import compute_stats, boxplot_accuracy
import CNN 
import WS 
import WS_AL 

# Setting the seed for reproducible results
torch.manual_seed(0)

# Models to compare
models = [CNN.ConvNet, WS.WS_Net, WS_AL.AuxLoss_Net]

# Statistics for each model (average and std)
stats, labels_bp = compute_stats(models, nb_rounds=15, nb_samples=1000, nb_epochs=50, mini_batch_size=100, criterion=nn.CrossEntropyLoss(), eta=.5, eta2=1e-2, alpha=.65, gamma=.75, lossplot=False)

# Display performances comparison (boxplot)
boxplot_accuracy(stats, labels_bp)