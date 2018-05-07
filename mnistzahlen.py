import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

kwargs = {"num_workers": 1, "pin_memory": True}
train_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download = True, transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])), batch_size=64, shuffle = True, **kwargs)

test_data = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, download = True, transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])), batch_size=64, shuffle = True, **kwargs)



class Netz(nn.Module):

    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1,5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 20, kernel_size = 5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = -1)
        print(x.size())
        


model = Netz()
model.cuda()



optimizer = optim.Adam(model.parameters(), lr = 0.01 )



def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print("epoche: {}, loss: {}".format(epoch, loss))


def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        data = Variable(data.cuda(), volatile = True)
        target = Variable(target.cuda())
        out = model(data)
        loss += F.nll_loss(out, target, size_average = False)
        prediction = out.data.max(1, keepdim = True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    loss = loss / len(test_data.dataset)
    print("Durchsnittsloss: {}".format(loss))
    print("Genauigkeit: ", 100.*correct/len(test_data.dataset))
    


for epoch in range(1, 10):
    train(epoch)
    test()