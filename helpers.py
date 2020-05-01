import statistics
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#####################################################################

def normalize(train_input, test_input):
    '''
    normalizes train_input and test_input and returns them
    '''
    mu, std = train_input.mean(), train_input.std()
    
    train_input_norm = train_input.sub_(mu).div_(std)
    test_input_norm = test_input.sub_(mu).div_(std)
    
    return train_input_norm, test_input_norm

#####################################################################

def plot_loss_acc(train_loss, test_accuracy):
    '''
    creates plot with train loss and test accuracy
    '''
    fig, ax1 = plt.subplots(figsize=(10,7))

    # plot train loss with log axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train loss')
    ax1.set_yscale("log")
    plt1 = ax1.plot(range(len(train_loss)), train_loss, color=color, label='Train loss')
    ax1.tick_params(axis='y')

    # create second axis
    ax2 = ax1.twinx()  

    # plot test accuracy on the same plot 
    color = 'tab:blue'
    ax2.set_ylabel('Test accuracy %')  
    plt2 = ax2.plot(range(len(test_accuracy)), test_accuracy, color=color, label='Test accuracy')
    ax2.tick_params(axis='y')

    # set legends
    plts = plt1+plt2
    labs = [p.get_label() for p in plts]
    plt.legend(plts, labs, loc='center right')

    fig.tight_layout() 
    plt.show()

    # save plot
    plt.savefig('trainloss_testacc.png', dpi=600)

######################################################################

def boxplot_accuracy(losses):
    '''
    creates boxplots of the train and test accuracies of all the models using pandas and seaborn
    '''
    
    # create pandas dataframe with losses and train/test column
    acc = {'CNN train':losses[0], 'CNN test':losses[1], 'WS train':losses[2], 'WS test':losses[3], 'WS&AuxLoss train':losses[4], 'WS&AuxLoss test':losses[5]}
    df_acc = pd.DataFrame(data=acc)
    df_acc_melt = pd.melt(df_acc)
    df_acc_melt['train'] = [1 if x[-1]=='n' else 0 for x in df_acc_melt['variable']]
    colors = {train: sns.xkcd_rgb["light blue"] if train == 1 else sns.xkcd_rgb["pale red"] for train in df_acc_melt.train}
    
    # draw boxplots 
    bxplt = sns.boxplot(x="variable", y="value", hue="train", data=df_acc_melt, palette=colors, dodge=False)
    bxplt.legend_.remove()
    fig = plt.gcf()
    fig.set_size_inches(14, 10)
    plt.ylabel('Accuracy %', size = 14)
    plt.xlabel('Networks', size = 14)
    plt.tick_params(labelsize=12)
    sns.set_style("whitegrid")

    # save boxplots 
    plt.savefig('boxplot_acc.png', dpi=600)