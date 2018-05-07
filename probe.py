import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

network = MyNetwork()
network = network.cuda()
print(network)

if os.path.isfile("meinNetz.pth"):
    netz = torch.load("meinNetz.pth")


for i in range(100):
    x = [1,0,0,0,1,0,0,0,1,1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    input = input.cuda()
    print(input)
    #input = Variable(x)

    out = network(input)

    x = [0,1,1,1,0,1,1,1,0,0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    target = target.cuda()

    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(loss)
    print(loss.grad_fn.next_functions[0][0])

    network.zero_grad()
    loss.backward()

    optimizer = optim.SGD(network.parameters(), lr=0.1)
    optimizer.step()

torch.save(network, "meinNetz.pth")