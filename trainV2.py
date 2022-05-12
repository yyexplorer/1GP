
import torch
import yy 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
from torch.nn import functional as F
import matplotlib.pyplot as plt
from itertools import cycle
import time
'''
相比于v1  提高了深度
'''


yy.tools.setup_seed(0)
outputONNXPath='C:/Users/yuanying/Desktop/testonnx.onnx'
dataPath='D:/1Datasets/MoVi'
# fileName='MoViData_Subj_86_87_Re_Tr_Inte_10.npy'
fileName='MoViData_Subj_46_47_50_55_60_65_70_71_75_80_81_82_83_84_85_88_89_90_Re_Tr_Inte_10.npy'
epochSize = 5
batchSize = 50
learningRate = 0.1
picture=yy.drawSomeLines(2)

#网络
net=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),#这之后与另一支合并
    yy.myNN.ResNetBlock(64,128,2),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,256,2),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,512,2),
    yy.myNN.ResNetBlock(512,512),
    yy.myNN.ResNetBlock(512,512),
    yy.myNN.ResNetBlock(512,256),
    yy.myNN.ResNetBlock(256,128),
    yy.myNN.ResNetBlock(128,64),
    yy.myNN.ResNetBlock(64,32),
    yy.myNN.ResNetBlock(32,20),
    # yy.myNN.relu2d(size=(10,13)),
    nn.Flatten(2),
    nn.Linear(130,81),
    nn.ReLU(),
    nn.Linear(81,27),
    nn.ReLU(),
    nn.Linear(27,9),
    nn.ReLU(),
    nn.Linear(9,3),
    nn.Sigmoid(),
)


#测试
testx=torch.randn(size=(1,3,300,400),device='cuda',dtype=torch.float32)
net.to('cuda')
print('测试网络输入',testx.shape)
with torch.no_grad():
    print('测试网络输出',net(testx).shape)
# torch.onnx.export(net,torch.randn(size=(1,3,300,400)).to('cuda'),outputONNXPath)
# exit()

#训练数据
allVideo, allJoints=yy.dataProcess.loadYYDataToNumpy(fileName=fileName)
allJoints=yy.dataProcess.JointsNormalization(allJonits=allJoints)#归一化
amass52to20,amass20pt=yy.MoViConst.getAmass20()
allJoints=allJoints[:,amass52to20,:]#转为amass20
allVideo=torch.from_numpy(allVideo)
allJoints=torch.from_numpy(allJoints)
print('allVideo',allVideo.shape,allVideo.dtype)
print('allJoints',allJoints.shape,allJoints.dtype)
trainData=yy.myNN.tensorDataset(allVideo,allJoints)
trainData=DataLoader(trainData,batch_size=batchSize,shuffle=True)
print('训练数据 迭代器',type(trainData),len(trainData))
#评估数据
fileNametest='MoViData_Subj_86_87_Re_Tr_Inte_10.npy'
tv,tj=yy.dataProcess.loadYYDataToNumpy(fileName=fileNametest)
tj=yy.dataProcess.JointsNormalization(allJonits=tj)#归一化
tj=tj[:,amass52to20,:]#转为amass20
tv=torch.from_numpy(tv)
tj=torch.from_numpy(tj)
evalData=yy.myNN.tensorDataset(tv,tj)
evalBatchSize=100
evalData=DataLoader(evalData,batch_size=evalBatchSize,shuffle=True)
evalData=cycle(evalData)

def mseAndCosineLoss(a,b,propotion=0.0001):
    mse=F.mse_loss(a,b)
    co=torch.sum(1-torch.cosine_similarity(a, b,dim=2))
    return torch.add(torch.mul(co,propotion),torch.mul(mse,1-propotion))


#正式训练
net.apply(yy.myNN.init_weights)
net.to('cuda')
optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
# optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
# optimizer = torch.optim.RMSprop(net.parameters(), lr=learningRate)
# loss=F.mse_loss
# loss=F.l1_loss
loss= mseAndCosineLoss

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
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        yl=l.cpu().detach().numpy().tolist()#用于输出训练集的loss
        ##评估
        ex,ey=next(evalData)
        ex=(ex/255).float()
        ex,ey=ex.to('cuda'),ey.to('cuda')
        with torch.no_grad():
            tyHat=net(ex)
            tl=loss(tyHat, ey)
            tl=tl.cpu().detach().numpy().tolist()
        picture.draw([epoch+i/num_batches,yl],[epoch+i/num_batches,tl])
        timer.stop()
    print('第',epoch,'个epoch训练完成')
print('共需要时间',timer.sum())

with torch.no_grad():
    tx,ty=next(iter(trainData))
    tx,ty=tx.to('cuda'),ty.to('cuda')
    tx=(tx/255).float()
    ex,ey=next(evalData)
    ex,ey=ex.to('cuda'),ey.to('cuda')
    ex=(ex/255).float()
    tyH=net(tx)
    eyH=net(ex)
    print('训练集误差',loss(ty, tyH))
    print('评估集误差',loss(ey, eyH))
# net.to('cpu')
torch.onnx.export(net,torch.randn(size=(1,3,300,400)).to('cuda'),outputONNXPath)
plt.show()
