import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

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
        self.bnconv1 = nn.BatchNorm2d(32)
        
        # self.conv2 : takes 32x6x6, gives 64x4x4, then maxpool(k=2) -> outputs 64x2x2 to the fc layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bnconv2 = nn.BatchNorm2d(64)
        
        # Gets in 64x2x2, convers to 1x250
        self.fc1 = nn.Linear(256,264)
        self.bn1 = nn.BatchNorm1d(264)
        
        # Second layer : 250 to 100
        self.fc2 = nn.Linear(264,100)  
        self.bn2 = nn.BatchNorm1d(100)
        
        # Output dim 10 so we can test the aux loss for classifying numbers
        # Use softmax on fc3 in final prediction layer?
        self.fc3 = nn.Linear(100,10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bnconv1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.bnconv2(self.conv2(x)), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x.softmax(1)

######################################################################
    
class final_predictionlayer(nn.Module):
    def __init__(self):
        super(final_predictionlayer,self).__init__()
        self.fc1 = nn.Linear(20,200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200,50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50,2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x.softmax(1)

######################################################################
    
class AuxLossBest_Net(nn.Module):
    def __init__(self):
        super(AuxLossBest_Net,self).__init__()
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
        
        return x, tmp1, tmp2
    