import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.utils.tensorboard

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1=[]
        for i in range(2):
            self.layer1.append(nn.Linear(3, 1))
        self.fc21 = nn.Linear(2, 1)

    def forward(self, x):
        x1 = F.relu(self.layer1[0](x[:,:3]))
        x2 = F.relu(self.layer1[1](x[:,3:]))
        x = torch.sigmoid((self.fc21(torch.cat((x1,x2),dim=1))))
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layer1=nn.ModuleList()
        for i in range(2):
            self.layer1.append(nn.Linear(3, 1))
        self.fc21 = nn.Linear(2, 1)

    def forward(self, x):
        x1 = F.relu(self.layer1[0](x[:,:3]))
        x2 = F.relu(self.layer1[1](x[:,3:]))
        x = torch.sigmoid((self.fc21(torch.cat((x1,x2),dim=1))))
        return x

class Chanel(nn.Module):
    def __init__(self):
        super(Chanel, self).__init__()
        self.n1 = Net2()
        self.n2 = Net2()
        self.fc21 = nn.Linear(2, 1)

    def forward(self, x):
        x1 = self.n1(x)
        x2 = self.n2(x)
       # print(torch.cat((x1,x2),dim=0))
        x = torch.sigmoid(self.fc21(torch.cat((x1,x2),dim=1)))
        #print(x)
        return x


net = Net2()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

x=torch.unsqueeze(torch.Tensor([ 1,  2,  3, 45, 67, 45]),dim=0)
labels= torch.unsqueeze(torch.Tensor([ 0]),dim=0)

optimizer.zero_grad()
#print(torch.squeeze(net.n1.layer1[1].weight))
y=net(x)
loss = criterion(y, labels)
print(loss)
loss.backward()
optimizer.step()
print(torch.squeeze(net.layer1[1].weight))
print(net.layer1[1].weight.grad)
