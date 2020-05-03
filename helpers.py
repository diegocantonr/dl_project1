import torch
from torch import nn
from torch import optim

import CNN as CNN
import WS as WS
import WS_AL as WS_AL
import dlc_practical_prologue as prologue

import statistics

# Only used for plots 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

######################################################################

def normalize(train_input, test_input):
    '''
    Normalizes train_input and test_input 
    Returns the normalized inputs
    '''
    mu, std = train_input.mean(), train_input.std()
    
    train_input_norm = train_input.sub_(mu).div_(std)
    test_input_norm = test_input.sub_(mu).div_(std)
    
    return train_input_norm, test_input_norm

######################################################################

def get_data(nb_samples, device):
    '''
    Loads data using dlc_practical_prologue.py, normalizes them, sets autograd to True 
    and sends them to the correct device (GPU or CPU) depending on the computer
    Returns the data (input, target, classes)
    '''
    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb_samples)

    # Normalize data
    train_input, test_input = normalize(train_input, test_input)
    
    # Set autograd to True
    train_input, train_target, train_classes = train_input.float().requires_grad_(), train_target.float().requires_grad_(), train_classes.float().requires_grad_()
    test_input, test_target, test_classes = test_input.float().requires_grad_(), test_target.float().requires_grad_(), test_classes.float().requires_grad_()
    
    # Send tensor to CPU or GPU
    train_input = train_input.to(device)
    test_input = test_input.to(device)
    train_target = train_target.to(device)
    test_target = test_target.to(device)
    train_classes = train_classes.to(device)
    test_classes = test_classes.to(device)

    return train_input, test_input, train_target, test_target, train_classes, test_classes

######################################################################

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    # Written by Francois Fleuret <francois@fleuret.org>
    '''
    Computes nb of classification errors given a model, a data input and the targets
    Return the obtained number of errors 
    '''
    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if (str(type(model)) == "<class 'CNN.ConvNet3'>" or str(type(model)) == "<class 'WS.WS_Best_Net'>"):
            output = model(data_input.narrow(0, b, mini_batch_size))

        if (str(type(model)) == "<class 'WS_AL.AuxLossBest_Net'>"):
            output = model(data_input.narrow(0, b, mini_batch_size))[0]
        
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors += + 1

    return nb_errors

######################################################################

def train_model(model, train_input, test_input, train_target, test_target, train_classes, test_classes, mini_batch_size=100, nb_epochs=25, criterion=nn.CrossEntropyLoss(), eta=9e-2, eta2=1e-3, alpha=.75, gamma=1): 
    '''
    Trains a given model using given datasets, mini_batch_size, nb_epochs, the loss used (criterion) and parameters gamma, alpha and eta
    Returns the obtained losses and accuracies 
    '''
    print('> Currently training:', model.__class__.__name__)

    # Squeeze the classes labels (hotlabeling) for the auxLoss
    trainlabel_1 = (train_classes.narrow(1,0,1)).squeeze()
    trainlabel_2 = (train_classes.narrow(1,1,1)).squeeze()

    losses = []
    accuracies = []
    
    for e in range(nb_epochs):
        
        # Change learning rate after 3/5 of the epochs (-> larger at the beginning, smaller at the end)
        if e < int(nb_epochs*(3/5)):
            optimizer = optim.SGD(model.parameters(), lr = eta, weight_decay=5e-4)
        else: 
            optimizer = optim.SGD(model.parameters(), lr = eta2, weight_decay=5e-4)
        
        sum_loss = 0
        for b in list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(train_input.size(0))), batch_size=mini_batch_size, drop_last=False)):
            
            # If the model is a simple CNN or a CNN with weight sharing compute model output, loss and optimize this way 
            if (str(type(model)) == "<class 'CNN.ConvNet3'>" or str(type(model)) == "<class 'WS.WS_Best_Net'>"):
                output = model(train_input[b])
                loss = criterion(output, train_target[b].long())
                model.zero_grad()
                loss.backward() 
                optimizer.step()
                sum_loss += loss.item()

            # If the model is a CNN with weight sharing and an auxiliary loss compute model output, compoute the different losses and optimize that way
            if (str(type(model)) == "<class 'WS_AL.AuxLossBest_Net'>"):
                out_compare, out_1, out_2 = model(train_input[b])
                
                #Main Loss
                loss_compare = criterion(out_compare, train_target[b].long())
                
                #AuxLoss
                loss_1 = criterion(out_1, trainlabel_1[b].long())
                loss_2 = criterion(out_2, trainlabel_2[b].long())
                
                #Weighted sum
                loss = alpha*loss_1 + alpha*loss_2 + gamma*loss_compare
                
                model.zero_grad()
                loss.backward()
                optimizer.step() 
                
                sum_loss += loss.item()
        
        losses.append(sum_loss)

        test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        test_acc = 100-100*(test_error/test_input.size(0))
        accuracies.append(test_acc)
        
    return losses, accuracies

######################################################################

def run_model(model_in, device, nb_samples=1000, nb_epochs=25, mini_batch_size=100, criterion=nn.CrossEntropyLoss(), eta=9e-2, eta2=1e-3, alpha=.75, gamma=1, lossplot=True):
    '''
    Given a model, and the needed parameters, loads the data, moves model to the appropriate device (GPU or CPU), 
    trains model, computes accuracy and plots train loss + test accuracy
    Returns the train and test accuracies and the train loss
    '''

    # Load and prepare data
    train_input, test_input, train_target, test_target, train_classes, test_classes = get_data(nb_samples, device)

    # Initialize weights and move model to appropriate device
    model = model_in()
    model.to(device)
    model.train(True)

    # Train model and return list of loss at each epoch
    train_loss, accuracies = train_model(model, train_input, test_input, train_target, test_target, train_classes, test_classes, mini_batch_size, nb_epochs, criterion.to(device), eta, eta2, alpha, gamma)
    
    model.train(False)
    
    # Compute accuracy from nb of errors
    train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) 
    train_acc = 100-100*(train_error/nb_samples)
    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    test_acc = 100-100*(test_error/nb_samples)
    
    # Plot train loss and test accuracy over all epochs if lossplot=True
    if lossplot:
        plot_loss_acc(train_loss, accuracies)
    
    return train_acc, test_acc, train_loss

######################################################################

def compute_stats(models, nb_rounds=10, nb_samples=1000, nb_epochs=25, mini_batch_size=100, criterion=nn.CrossEntropyLoss(), eta=9e-2, eta2=1e-3, alpha=.75, gamma=1, boxplot=True, lossplot=True):
    '''
    Given a list of models and the number of rounds to perform and other needed parameters, computes the average train and test errors + accuracies
    Returns the average errors of train and test sets over the rounds
    '''

    # Verify if GPU is available otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))

    avg_errors = [[[] for x in range(2)] for y in range(len(models))]

    for e in range(nb_rounds):
        for ind, mod in enumerate(models):
            
            # For each model of the list call the function run (trains + computes accuracy and loss + plots results)
            train_acc, test_acc, _ = run_model(mod, device, nb_samples, nb_epochs, mini_batch_size, criterion, eta, eta2, alpha, gamma, lossplot=lossplot)
            
            # Store average errors of train and test
            avg_errors[ind][0].append(train_acc)
            avg_errors[ind][1].append(test_acc)
            
            print(e, mod().__class__.__name__+': train error = {:0.2f}%,  test error = {:0.2f}%'.format(100-train_acc, 100-test_acc))
    
    #labels for boxplot
    labels_bp = []
    
    # Compute performance estimates over nb_rounds (average + std deviation)
    
    print("\n", '> Perfromance estimates :')
    
    for ind, mod in enumerate(models):
        name = mod().__class__.__name__
        
        if nb_rounds > 1:
            # Compute averages of train and test errors 
            train_average = statistics.mean(avg_errors[:][ind][0])
            test_average = statistics.mean(avg_errors[:][ind][1])
            
            # Compute standard deviation of train and test errors
            train_std = statistics.stdev(avg_errors[:][ind][0])
            test_std = statistics.stdev(avg_errors[:][ind][1])
        
        else: # nb_rounds = 1 
            # Compute averages of train and test errors 
            train_average = avg_errors[ind][0][0]
            test_average = avg_errors[ind][1][0]
            
            # Compute standard deviation of train and test errors
            train_std = 0
            test_std = 0
             
        print(name +' : train_acc average = {:0.2f}%, test_acc average = {:0.2f}%'.format(train_average, test_average))
        print(name +' : train_acc std = {:0.2f}%, test_acc std = {:0.2f}%'.format(train_std, test_std))
        
        labels_bp.append(name)
        
    return avg_errors, labels_bp

######################################################################

def run(models, nb_rounds, nb_epochs, boxplot=True, lossplot=False):
    '''
    For all models of the list models and given the nb_rounds and nb_epochs, will run, 
    train, compute stats of the models and plot the boxplot of the obtained accuracies if boxplot=True
    '''
    avg_errors, labels_bp = compute_stats(models, nb_rounds=nb_rounds, nb_epochs=nb_epochs, lossplot=lossplot)
    
    if (boxplot):
        boxplot_accuracy(avg_errors, labels_bp)

######################################################################

def plot_loss_acc(train_loss, test_accuracy):
    '''
    Creates plot with the train loss and test accuracy
    Saves the obtained plot
    '''
    fig, ax1 = plt.subplots(figsize=(8,8))

    # Plot train loss with log axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train loss')
    ax1.set_yscale("log")
    plt1 = ax1.plot(range(len(train_loss)), train_loss, color=color, label='Train loss')
    ax1.tick_params(axis='y')

    # Create second axis
    ax2 = ax1.twinx()  

    # Plot test accuracy on the same plot 
    color = 'tab:blue'
    ax2.set_ylabel('Test accuracy %')  
    plt2 = ax2.plot(range(len(test_accuracy)), test_accuracy, color=color, label='Test accuracy')
    ax2.tick_params(axis='y')

    # Set legends
    plts = plt1+plt2
    labs = [p.get_label() for p in plts]
    plt.legend(plts, labs, loc='center right')

    fig.tight_layout() 

    # Save plot
    plt.savefig('trainloss_testacc.png', dpi=600, transparent=True)
    plt.show()
    
#####################################################################

def boxplot_accuracy(losses, labels):
    '''
    Plots all boxplots of the train and test accuracies of all tested models using pandas and seaborn 
    Saves the obtained plot
    '''
    # Create pandas dataframe with losses and train/test column
    acc = {}
    for ind, lab in enumerate(labels):
        tmp = {lab+' train':losses[ind][0], lab+' test':losses[ind][1]}
        acc.update(tmp)
      
    df_acc = pd.DataFrame(data=acc)
    df_acc_melt = pd.melt(df_acc)
    df_acc_melt['train'] = [1 if x[-1]=='n' else 0 for x in df_acc_melt['variable']]
    colors = {train: sns.xkcd_rgb["light blue"] if train == 1 else sns.xkcd_rgb["pale red"] for train in df_acc_melt.train}
    
    # Draw boxplots 
    sns.set_style("whitegrid", {'ytick.left': True, 'axes.edgecolor': 'k'})
    bxplt = sns.boxplot(x="variable", y="value", hue="train", data=df_acc_melt, palette=colors, dodge=False)
    bxplt.legend_.remove()
    fig = plt.gcf()
    fig.set_size_inches(20, 12 )
    plt.ylabel('Accuracy %', size = 14)
    plt.xlabel('Networks', size = 14)
    plt.tick_params(labelsize=12)

    # Save boxplots 
    plt.savefig('boxplot_acc.png', dpi=600, transparent=True)
    plt.show()