import torch
from torch import nn
from helpers import model_tuning
import WS_AL

# Model to tune
model = WS_AL.AuxLoss_Net

# Parameters grid
params = {
    'eta' : [5e-1, 2e-1, 9e-2, 5e-2, 1e-2, 5e-3],
    'eta2' : [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 8e-5],
    'alpha' : [.2, .5, .65, .75, .9, 1],
    'gamma' : [.2, .5, .65, .75, .9, 1]
}

# Tuning with the grid search method - for each combination of parameters, model is tested and averaged over 10 rounds on a validation set - returns the best combination of parameters
best_params = model_tuning(model, params, validation_size=0.2, nb_rounds=10, nb_samples=1000, nb_epochs=50, mini_batch_size=100, criterion=nn.CrossEntropyLoss())

print('The best hypers-parameters for the model are :', best_params)