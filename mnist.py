from torchvision import datasets, transforms
import torch.utils.data
import torch

import torch.nn as nn
import torch.optim as optim

# 数据准备
train_data = datasets.MNIST(
                            root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False,
                            )

test_data = datasets.MNIST(
                            root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=False,
                            )

input_size = 28
num_classes = 10
num_epochs = 3
batch_size = 64

train_loader = torch.utils.data.DataLoader(
                                            batch_size=batch_size,
                                            shuffle=True,
                                            dataset=train_data
                                          )
test_loader = torch.utils.data.DataLoader(
                                            batch_size=batch_size,
                                            shuffle=True,
                                            dataset=test_data
                                          )

class Cnn(nn.Module):

    # 前提准备
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(


            # 1*28*28
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 16*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 32*7*7
        )
        self.linear= nn.Linear(32*7*7,10)

    # 向前传递
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        out = self.linear(x)

        return out

# 准确率函数
def accuracy(prediction, lables):
    pred = torch.max(prediction.data,1)[1]
    right = pred.eq(lables.data.view_as(pred)).sum()

    return right, len(lables)

# 实例化
net = Cnn()
# 优化器选择Adam
optimzier = optim.Adam(net.parameters(),lr=0.001)
# 损失函数选择交叉熵函数
criterion = nn.CrossEntropyLoss()

# 训练开始
for epoch in range(num_epochs):
    train_rights = []

    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        output = net(data)
        print(output)
        train_rights_ = []

        for x in output:
            temp = torch.argmax(x,keepdim=False).numpy()
            number = int(temp)

            train_rights_.append(number)
        print(train_rights_)
        print(target)
        loss = criterion(output, target)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx%100 == 0:
            net.eval()
            val_rights = []

            for (data,target) in test_loader:
                output = net(data)
                # print(output)
                right = accuracy(output,target)
                # print(right)
                val_rights.append(right)


            # 计算准确率
            train_r = (sum([top[0] for top in train_rights]),
                       sum([top[1] for top in train_rights]))
            # print(train_r)
            val_r = (sum([top[0] for top in val_rights]),
                       sum([top[1] for top in val_rights]))

            print('当前epoch:{} [{}/{} ({:.0f})%]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试机正确率：{:.6f}%'.format(
                epoch,batch_idx*batch_size,len(train_loader.dataset),
                100.*batch_idx/len(train_loader),
                loss.data,
                100.*train_r[0].numpy() / train_r[1],
                100.*val_r[0].numpy() / val_r[1]
            ))






















