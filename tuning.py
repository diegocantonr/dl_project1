import torch
from torch import nn
from helpers import model_tuning
import WS_AL as WS_AL

model = WS_AL.AuxLoss_Net

params = {
    'eta' : [5e-1, 2e-1, 1e-1, 9e-2, 7e-2, 5e-2, 3e-2, 1e-2, 5e-3],
    'eta2' : [3e-2, 1e-2, 5e-3, 1e-3, 8e-4, 5e-4, 1e-4, 8e-5, 5e-5],
    'alpha' : [.2, .4, .5, .6, .65, .7, .75, .8, .85, .9, 1],
    'gamma' : [.2, .4, .5, .6, .65, .7, .75, .8, .85, .9, 1]
}

best_params = model_tuning(model, params, validation_size=0.2, nb_rounds=10, nb_samples=1000, nb_epochs=50, mini_batch_size=100, criterion=nn.CrossEntropyLoss())

print('The best hypers-parameters for the model are :', best_params)