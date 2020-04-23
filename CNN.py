import helpers
from helpers import*
import statistics

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x.softmax(1)
    
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.softmax(1)

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 2)
        
        self.dropout = nn.Dropout(0.20)

    def forward(self, x):                                                     
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))      
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))      
        x = self.dropout(F.relu(self.fc1(x.view(-1, 512))))   
        x = self.fc2(x)                                                      
        return x.softmax(1)
    
class ConvNet4(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x.softmax(1)
    
class ConvNet5(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x.softmax(1)

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs, criterion, eta=9e-2): 
    optimizer = optim.SGD(model.parameters(), lr = eta)
    losses = []
    
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
    
    return losses

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors += + 1

    return nb_errors

def run_model(model, nb_samples, nb_epochs, mini_batch_size, criterion=nn.CrossEntropyLoss()):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))
    
    # Load data
    train_input, train_target, _, test_input, test_target, _ = prologue.generate_pair_sets(nb_samples)

    # Standardize data
    train_input = standardize(train_input)
    test_input = standardize(test_input)
    
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
   
    train_loss = train_model(model, train_input, train_target, mini_batch_size, nb_epochs, criterion.to(device), eta=2e-1)
    
    model.train(False)
    
    train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) 
    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    
    return train_error, test_error, train_loss

def compute_stats(nb_average=10, nb_samples=1000, nb_epochs=25, mini_batch_size=100):
    
    models = [ConvNet3]
    avg_errors = [[[] for x in range(2)] for y in range(len(models))]

    for e in range(nb_average):
        for ind, mod in enumerate(models):
            train_error, test_error, _ = run_model(mod, nb_samples, nb_epochs, mini_batch_size)
            
            avg_errors[ind][0].append(train_error)
            avg_errors[ind][1].append(test_error)
            
            print(e, mod().__class__.__name__+': train error Net {:0.2f}%'.format((100 * train_error) / nb_samples))
            print(e, mod().__class__.__name__+': test error Net {:0.2f}%'.format((100 * test_error) / nb_samples))
            
    for ind, mod in enumerate(models):
        train_average = statistics.mean(avg_errors[:][ind][0])
        test_average = statistics.mean(avg_errors[:][ind][1])
        print(mod().__class__.__name__+' : train_error average = {:0.2f}%, test_error average = {:0.2f}%'.format((100 * train_average) / nb_samples, (100 * test_average) / nb_samples))
    
    return avg_errors