from helpers import *

######################################################################

models = [CNN.ConvNet3, WS.WS_Best_Net, WS_AL.AuxLossBest_Net]

run(models, nb_rounds=15, nb_epochs=50, boxplot=True, lossplot=False)


