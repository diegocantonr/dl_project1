import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

######################################################################

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        
        # Conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        
        # FC
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):                                                     
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))      
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))      
        x = self.dropout(F.relu(self.fc1(x.view(-1, 512))))   
        x = self.fc2(x)  
        
        return x.softmax(1)

######################################################################
    
class shared_layers(nn.Module):
    '''
    This Module contains all layers with shared weights
    2 conv layers and 3 hidden layers
    '''
    def __init__(self):
        super(shared_layers, self).__init__()
        # self.conv1 : takes 1x14x14, gives 32x12x12, then maxpool(k=2) -> 32x6x6
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        
        # self.conv2 : takes 32x6x6, gives 64x4x4, then maxpool(k=2) -> outputs 64x2x2 to the fc layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Gets in 64x2x2, converts to 1x250
        self.fc1 = nn.Linear(256, 200)
        self.bn1 = nn.BatchNorm1d(200)
        
        # Second layer : 200 to 100
        self.fc2 = nn.Linear(200, 100)  
        self.bn2 = nn.BatchNorm1d(100)
        
        # Outputs dim 10 so we can test the aux loss for classifying numbers
        # Use softmax on fc3 in final prediction layer?
        self.fc3 = nn.Linear(100,10)
       
        self.dropout = nn.Dropout(0.2)
   
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x.softmax(1)
    
######################################################################

class final_predictionlayer(nn.Module):
    def __init__(self):
        super(final_predictionlayer,self).__init__()
        self.fc1 = nn.Linear(20,150)
        self.bn1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150,50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50,2)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        
        return x.softmax(1)
    
######################################################################

class WS_Best_Net(nn.Module):
    def __init__(self):
        super(WS_Best_Net, self).__init__()
        self.shared = shared_layers()
        self.final = final_predictionlayer()
    
    def forward(self,x):
        # View only one image
        tmp1 = x.narrow(1,0,1)
        # View only one image 
        tmp2 = x.narrow(1,1,1) 
        
        # Apply the conv layers
        tmp1 = self.shared(tmp1)
        tmp2 = self.shared(tmp2)
        
        # View and final prediction
        output = torch.cat((tmp1,tmp2),1)
        x = self.final(output)
        
        return x

######################################################################

class WS_net(nn.Module):
    def __init__(self):
        super(WS_net, self).__init__()
        self.ConvNet = ConvNet3()
        self.final = final_predictionlayer()
    
    def forward(self,x):
        # View only one image
        tmp1 = x.narrow(1,0,1) 
        # View only one image
        tmp2 = x.narrow(1,1,1) 
       
        # Apply the conv layers
        tmp1 = self.ConvNet(tmp1) 
        tmp2 = self.ConvNet(tmp2)
      
        # View and final prediction
        output = torch.cat((tmp1,tmp2),1)
        x = self.final(output)
        
        return x
    
