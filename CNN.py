import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

############################################################################################################################################

class ConvNet(nn.Module):
    ''' This model takes 2 images as input                 ->   2 x 14 x 14   Input
        and then the following operations are applied      ->  32 x 12 x 12   Convolution (3x3 kernel)
                                                           ->  32 x  6 x  6   MaxPool (2x2 kernel + stride=2) + ReLu
                                                           ->  64 x  4 x  4   Convolution (3x3 kernel)
                                                           ->  64 x  2 x  2   MaxPool (2x2 kernel + stride=2) + ReLu
        The model outputs 2 classes                        ->  256            view(-1)
        0 : if 1st digit is lesser of equal than the 2nd   ->  200            Fully connected layer 1 (256 -> 200) + ReLu + BatchNorm1d + Dropout
        1 : otherwise                                      ->  100            Fully connected layer 2 (200 -> 100) + ReLu + BatchNorm1d + Dropout
                                                           ->  2              Fully connected layer 3 (100 ->   2) = Output
    '''
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)  
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
       
        self.dropout = nn.Dropout(0.2)
   
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.dropout(self.bn1(F.relu(self.fc1(x.view(-1, 256)))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x