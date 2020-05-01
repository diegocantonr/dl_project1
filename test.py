import CNN as CNN
import WS as WS
import WS_AuxLoss as WS_AL


loss_cnn = CNN.compute_stats(nb_average=15, nb_epochs=50, lossplot=False)

loss_ws = WS.compute_stats(nb_average=15, nb_epochs=50, lossplot=False)

loss_ws_al = WS_AL.compute_stats(nb_average=15, nb_epochs=50, lossplot=False)

data = [loss_cnn[0][0], loss_cnn[0][1], loss_ws[0][0], loss_ws[0][1], loss_ws_al[0][0], loss_ws_al[0][1]]

