from helpers import*
from plots import *
import statistics
   
class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.20)

    def forward(self, x):                                                     
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))      
        x = self.dropout(F.relu(self.fc1(x.view(-1, 512))))   
        x = self.fc2(x)                                                      
        return x.softmax(1)
    
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
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = F.softmax(self.fc3(x),dim=1)
        return x
    
class final_predictionlayer(nn.Module):
    def __init__(self):
        super(final_predictionlayer,self).__init__()
        self.fc1 = nn.Linear(20,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x.softmax(1)
    
class WS_Best_Net(nn.Module):
    def __init__(self):
        super(WS_Best_Net, self).__init__()
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
        return x

class WS_net(nn.Module):
    def __init__(self):
        super(WS_net, self).__init__()
        self.ConvNet = ConvNet3()
        self.final = final_predictionlayer()
    
    def forward(self,x):
        tmp1 = x.narrow(1,0,1) #viewing only one image
        tmp2 = x.narrow(1,1,1) #viewing only one image
       
        #applying the conv layers
        tmp1 = self.ConvNet(tmp1) 
        tmp2 = self.ConvNet(tmp2)
      
        #viewing and final prediction
        output = torch.cat((tmp1,tmp2),1)
        x = self.final(output)
        return x.softmax(1)
    
def train_model(model, train_input, train_target, test_input, test_target, mini_batch_size, nb_epochs, criterion, eta=9e-2): 
    optimizer = optim.SGD(model.parameters(), lr = eta, weight_decay=5e-4)
    losses = []
    accuracies = []
    
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).long())
            model.zero_grad()
            loss.backward() #backward pass 
            optimizer.step()
            sum_loss += loss.item()
        
        losses.append(sum_loss)
        
        test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        test_acc = 100-100*(test_error/test_input.size(0))
        accuracies.append(test_acc)
    
    return losses, accuracies

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors += + 1

    return nb_errors

def run_model(model, nb_samples, nb_epochs, mini_batch_size, device, criterion=nn.CrossEntropyLoss(), lossplot=True):
    
    # Load data
    train_input, train_target, _, test_input, test_target, _ = prologue.generate_pair_sets(nb_samples)

    # Normalize data
    train_input, test_input = normalize(train_input, test_input)
    
    # Set autograd to True
    train_input, train_target = train_input.float().requires_grad_(), train_target.float().requires_grad_()
    test_input, test_target = test_input.float().requires_grad_(), test_target.float().requires_grad_()
    
    # Send tensor to CPU or GPU
    train_input = train_input.to(device)
    test_input = test_input.to(device)
    train_target = train_target.to(device)
    test_target = test_target.to(device)
    
    model = model()
    model.to(device)
    model.train(True)
    
    # Train model and return list of loss at each epoch
    train_loss, accuracies = train_model(model, train_input, train_target, test_input, test_target, mini_batch_size, nb_epochs, criterion.to(device), eta=9e-2)
                            
    model.train(False)
    
    # compute accuracy from nb of errors
    train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) 
    train_acc = 100-100*(train_error/nb_samples)
    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    test_acc = 100-100*(test_error/nb_samples)
    
    # Plot train loss and test accuracy over all epochs if lossplot=True
    if lossplot:
        plot_loss_acc(train_loss, accuracies)
    
    return train_acc, test_acc, train_loss

def compute_stats(nb_average=10, nb_samples=1000, nb_epochs=25, mini_batch_size=100, lossplot=True):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))
    
    models = [WS_net, WS_Best_Net]
    avg_errors = [[[] for x in range(2)] for y in range(len(models))]

    for e in range(nb_average):
        for ind, mod in enumerate(models):
            train_acc, test_acc, _ = run_model(mod, nb_samples, nb_epochs, mini_batch_size, device, lossplot=lossplot)
            
            avg_errors[ind][0].append(train_acc)
            avg_errors[ind][1].append(test_acc)
            
            print(e, mod().__class__.__name__+': train error = {:0.2f}%,  test error = {:0.2f}%'.format(100-train_acc, 100-test_acc))
            
    for ind, mod in enumerate(models):
        train_average = statistics.mean(avg_errors[:][ind][0])
        test_average = statistics.mean(avg_errors[:][ind][1])
        
        train_std = statistics.stdev(avg_errors[:][ind][0])
        test_std = statistics.stdev(avg_errors[:][ind][1])
       
        print(mod().__class__.__name__+' : train_acc average = {:0.2f}%, test_acc average = {:0.2f}%'.format(train_average, test_average))
        print(mod().__class__.__name__+' : train_acc std = {:0.2f}%, test_acc std = {:0.2f}%'.format(train_std, test_std))
        
    return avg_errors