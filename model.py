import torch 
import torch.nn as nn
from data_preprocessing import DataSet
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*106*106,2)
        
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output))) 
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output
        
# dataset = DataSet(root_dir = '/media/oscar/Seagate/machine_learning_engineer/CNNs/data/')
# train_loader, test_loader = dataset.get_loader()
# model = SimpleCNN()
# # print(len(train_loader))
# for idx, dataset in enumerate(train_loader):
#     X = dataset[0]
#     y = dataset[1]
    
#     if idx < 1:
#         # print(idx,X.shape,y.shape)
#         model(X)
#     else:
#         break
