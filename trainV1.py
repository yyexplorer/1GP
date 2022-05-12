
from unittest import TestLoader
import torch
import yy 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import d2l.torch as d2l
from torch.nn import functional as F
import matplotlib.pyplot as plt
yy.tools.setup_seed(0)
d2l.Animator
outputONNXPath='C:/Users/yuanying/Desktop/testonnx.onnx'
dataPath='D:/1Datasets/MoVi'
fileName='MoViData_Subj_86_87_Re_Tr_Inte_10.npy'
epochSize = 100
batchSize = 100
learningRate = 5
picture=yy.drawSomeLines(2)
allVideo, allJoints=yy.dataProcess.loadYYDataToNumpy(fileName=fileName)
# allVideo=(allVideo/255).astype('float32')#修改颜色格式
allJoints=allJoints/np.max(abs(allJoints))#归一化
allVideo=torch.from_numpy(allVideo)
allJoints=torch.from_numpy(allJoints)
print('allVideo',allVideo.shape,allVideo.dtype)
print('allJoints',allJoints.shape,allJoints.dtype)
#pytorch数据
# allVideo = torch.randn(4, 3)
# allJoints = torch.rand(4)

trainData=yy.myNN.tensorDataset(allVideo,allJoints)
trainData=DataLoader(trainData,batch_size=batchSize,shuffle=True)
print('训练数据 迭代器',type(trainData),len(trainData))

#测试
net=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=6,kernel_size=7,padding=3,stride=2),
    nn.BatchNorm2d(6),
    nn.ReLU(),
    yy.myNN.ResNetBlock(6,6),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5,padding=2,stride=2),
    nn.BatchNorm2d(12),
    nn.ReLU(),
    yy.myNN.ResNetBlock(12,12),
    nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5,padding=2,stride=2),
    nn.BatchNorm2d(24),
    nn.ReLU(),
    yy.myNN.ResNetBlock(24,24),

    nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,padding=1,stride=1),
    nn.BatchNorm2d(48),
    nn.ReLU(),
    yy.myNN.ResNetBlock(48,48),

    nn.Conv2d(in_channels=48,out_channels=52,kernel_size=5,padding=2,stride=2),
    nn.BatchNorm2d(52),
    nn.ReLU(),
    yy.myNN.ResNetBlock(52,52),

    nn.Conv2d(in_channels=52,out_channels=52,kernel_size=7,padding=3,stride=1),
    nn.BatchNorm2d(52),
    nn.ReLU(),
    yy.myNN.ResNetBlock(52,52),
    yy.myNN.ResNetBlock(52,52),
    nn.Flatten(start_dim=2),
    nn.Linear(130,65),
    nn.BatchNorm1d(52),
    nn.ReLU(),
    nn.Linear(65,30),
    nn.BatchNorm1d(52),
    nn.ReLU(),
    nn.Linear(30,3),
    nn.BatchNorm1d(52),
    nn.Sigmoid(),
)
net.apply(yy.myNN.init_weights)
testx=allVideo[[1,2],...]
testy=allJoints[[1,2],...]
print('测试矩阵',testx.shape,testy.shape)
testx=(testx/255).float()
with torch.no_grad():
    print('测试输出',net(testx).shape)
    yHatBefore=net(testx)
#正式训练
net.to('cuda')
optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
loss = nn.CrossEntropyLoss()
num_batches =  len(trainData)
timer=yy.Timer()

for epoch in range(epochSize):
    net.train()
    for i, (X, y) in enumerate(trainData):
        X=(X/255).float()
        timer.start()
        optimizer.zero_grad()
        X, y = X.to('cuda'), y.to('cuda')
        y_hat = net(X)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        yl=loss.cpu().detach().numpy().tolist()#用于输出训练集的loss

        picture.draw([epoch+i/num_batches,yl])
        timer.stop()
    print('第',epoch,'个epoch训练完成')
print('共需要时间',timer.sum())

with torch.no_grad():
    net.to('cpu')
    yHatAfter=net(testx)
    print('训练前误差',F.mse_loss(yHatBefore, testy))
    print('训练后误差',F.mse_loss(yHatAfter, testy))
torch.onnx.export(net,torch.randn(size=(1,3,300,400)),outputONNXPath)
plt.show()
