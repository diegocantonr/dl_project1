import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, test_accuracy):
    fig, ax1 = plt.subplots(figsize=(10,7))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train loss')
    ax1.set_yscale("log")
    plt1 = ax1.plot(range(len(train_loss)), train_loss, color=color, label='Train loss')
    ax1.tick_params(axis='y')
    #ax1.set_ylim(1,50)

    ax2 = ax1.twinx()  

    color = 'tab:blue'
    ax2.set_ylabel('Test accuracy %')  
    plt2 = ax2.plot(range(len(test_accuracy)), test_accuracy, color=color, label='Test accuracy')
    ax2.tick_params(axis='y')

    plts = plt1+plt2
    labs = [p.get_label() for p in plts]
    plt.legend(plts, labs, loc='center right')

    fig.tight_layout() 
    plt.show()


#def boxplots_acc(train_accs, test_accs):
    
