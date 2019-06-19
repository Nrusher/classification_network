import torch
import torch.nn as nn
import torch.nn.functional as func

class LeNet(nn.Module):
    def __init__(self,class_num):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.fc1 = nn.Linear(16*22*22, 240)
        self.fc1 = nn.Linear(16*53*53, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, class_num)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = LeNet(101)
    input = torch.Tensor(1,3,224,224)
    print(input.shape)
    net(input)
    # print(net)