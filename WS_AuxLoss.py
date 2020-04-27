import helpers
from helpers import *
import statistics
from plots import *


class shared_layers(nn.Module):
    """This Module contains all layers with shared weights
    2 conv layers and 3 hidden layers"""
    def __init__(self):
        super(shared_layers, self).__init__()
        #self.conv1 : takes 1x14x14, gives 32x12x12, then maxpool(k=2) -> 32x6x6
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        #self.conv2 : takes 32x6x6, gives 64x4x4, then maxpool(k=2) -> outputs 64x2x2 to the fc layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #gets in 64x2x2, convers to 1x250
        self.fc1 = nn.Linear(2*2*64,264)
        self.bn1 = nn.BatchNorm1d(264)
        #second layer : 250 to 100
        self.fc2 = nn.Linear(264,100)  
        self.bn2 = nn.BatchNorm1d(100)
        #outputs dim 10 so we can test the aux loss for classifying numbers
        #use softmax on fc3 in final prediction layer?
        self.fc3 = nn.Linear(100,10)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2,stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2,stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1,2*2*64)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = F.softmax(self.fc3(x),dim=1)
        return x
    
class final_predictionlayer(nn.Module):
    def __init__(self):
        super(final_predictionlayer,self).__init__()
        self.fc1 = nn.Linear(20,200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200,50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50,2)
        self.dropout = nn.Dropout(0.20)
        
    def forward(self,x):
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x
    
class AuxLossBest_Net(nn.Module):
    def __init__(self):
        super(AuxLossBest_Net,self).__init__()
        self.shared = shared_layers()
        self.final = final_predictionlayer()
    
    def forward(self,x):
        tmp1 = x.narrow(1,0,1) #viewing only one image
        tmp2 = x.narrow(1,1,1) #viewing only one image
        
        #applying the conv layers
        tmp1 = self.shared(tmp1)
        tmp2 = self.shared(tmp2)
        
        #viewing and final prediction
        output = torch.cat((tmp1,tmp2),1)
        x = self.final(output)
        return x, tmp1, tmp2
    
def train_model(model, train_input, train_target, train_classes, test_input, test_target, mini_batch_size, nb_epochs, criterion, eta=9e-2, alpha=.75, gamma=1): 
    
    #Squeeze the classes labels (hotlabeling) for the auxLoss
    trainlabel_1 = (train_classes.narrow(1,0,1)).squeeze()
    trainlabel_2 = (train_classes.narrow(1,1,1)).squeeze()

    optimizer = optim.SGD(model.parameters(),lr=eta)
    losses = []
    accuracies = []

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):    
            
            out_compare, out_1, out_2 = model(train_input.narrow(0, b, mini_batch_size))
            #Main Loss
            loss_compare = criterion(out_compare, train_target.narrow(0, b, mini_batch_size).long())
            #AuxLoss
            loss_1 = criterion(out_1, trainlabel_1.narrow(0, b, mini_batch_size).long())
            loss_2 = criterion(out_2, trainlabel_2.narrow(0, b, mini_batch_size).long())
            #Weighted sum. Used to be Alpha*Loss1 + Beta*Loss2 + Gamma* Loss compare
            #Didn't work well, try again with other alpha/betas < 1.
            loss = alpha*loss_1 + alpha*loss_2 + gamma*loss_compare
            
            model.zero_grad()
            loss.backward()
            optimizer.step() 
            sum_loss += loss.item()
        
        acc = 1-(compute_nb_errors(model, test_input.narrow(0, b, mini_batch_size), test_target.narrow(0, b, mini_batch_size).long(), mini_batch_size)/mini_batch_size)
        accuracies.append(acc)

        losses.append(sum_loss)
        
    return losses, accuracies

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output,_,_ = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors += + 1

    return nb_errors

def run_model(model, nb_samples, nb_epochs, mini_batch_size, criterion=nn.CrossEntropyLoss(), lossplot=True):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))
    
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
    
    model = model()
    model.to(device)
    model.train(True)
    
    train_loss, accuracies = train_model(model, train_input, train_target, train_classes, test_input, test_target, mini_batch_size, nb_epochs, criterion.to(device), eta=9e-2, alpha=.75, gamma=1)
    
    model.train(False)
    
    train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) 
    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)

    # Plot train loss and test accuracy over all epochs if lossplot=True
    if lossplot:
        plot_loss_acc(train_loss, accuracies)
    
    return train_error, test_error, train_loss

def compute_stats(nb_average=10, nb_samples=1000, nb_epochs=25, mini_batch_size=100, lossplot=True, boxplot=True):
    
    models = [AuxLossBest_Net]
    avg_errors = [[[] for x in range(2)] for y in range(len(models))]

    train_accs = []
    test_accs = []

    for e in range(nb_average):
        for ind, mod in enumerate(models):
            if lossplot:
                train_error, test_error, _ = run_model(mod, nb_samples, nb_epochs, mini_batch_size, lossplot=True)
            else:
                train_error, test_error, _ = run_model(mod, nb_samples, nb_epochs, mini_batch_size, lossplot=False)
            
            avg_errors[ind][0].append(train_error)
            avg_errors[ind][1].append(test_error)
            
            print(e, mod().__class__.__name__+': train error Net {:0.2f}%'.format((100 * train_error) / nb_samples))
            print(e, mod().__class__.__name__+': test error Net {:0.2f}%'.format((100 * test_error) / nb_samples))
            
        train_accs.append(1-train_error/nb_samples)
        test_accs.append(1-test_error/nb_samples)
    
    for ind, mod in enumerate(models):
        train_average = statistics.mean(avg_errors[:][ind][0])
        test_average = statistics.mean(avg_errors[:][ind][1])
        print(mod().__class__.__name__+' : train_error average = {:0.2f}%, test_error average = {:0.2f}%'.format((100 * train_average) / nb_samples, (100 * test_average) / nb_samples))

    if boxplot:    
        plt.boxplot(train_accs)
        plt.boxplot(test_accs)  

    return avg_errors
        
        
        
        
        