
from re import S
import torch
import yy 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
from torch.nn import functional as F
import matplotlib.pyplot as plt
from itertools import cycle
import time
import yydsntnn as dsntnn
'''
相比于v2   加入dsntnn   并使用热图进行训练
'''
yy.tools.setup_seed(0)
outputONNXPath='C:/Users/yuanying/Desktop/testonnx.onnx'
dataPath='D:/1Datasets/MoVi'
# fileName='MoViData_Subj_86_87_Re_Tr_Inte_10.npy'
fileName='MoViData_Subj_46_47_50_55_60_65_70_71_75_80_81_82_83_84_85_88_89_90_Re_Tr_Inte_10.npy'
epochSize = 5
batchSize = 30
learningRate = 0.0002
sigma_t=1
picture=yy.drawSomeLines(2,xlim=[0,epochSize])

#网络
class CoordRegressionNetwork(nn.Module):
    '''
    '''
    def __init__(self,n_locations):
        super().__init__()
        # self.fcn=nn.Conv2d(inChannels,inChannels,kernel_size=3,padding=1)
        self.hm_conv = nn.Conv2d(n_locations, n_locations, kernel_size=1, bias=False,groups=1)

    def forward(self, images):
        # 1. Run the images through our FCN
        #先运行满连接
        # fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        #将outChannels变为输出的点数量  使用1*1卷积
        unnormalized_heatmaps = self.hm_conv(images)
        # print('unnormalized_heatmaps信息',unnormalized_heatmaps.shape)
        # 3. Normalize the heatmaps
        #将普通卷积结果进行softmax处理
        heatmaps = dsntnn.d2_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps
'''
netV3_1=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,32),
    yy.myNN.ResNetBlock(32,32),
    yy.myNN.ResNetBlock(32,32),
    yy.myNN.ResNetBlock(32,20),
    yy.myNN.ResNetBlock(20,20),
    yy.myNN.ResNetBlock(20,20),
    CoordRegressionNetwork(20),
)
netV3_2=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    yy.myNN.ResNetBlock(64,128),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,256),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(256,64),
    yy.myNN.ResNetBlock(64,64),
    nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
    yy.myNN.ResNetBlock(64,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,512),
    yy.myNN.ResNetBlock(512,512),
    yy.myNN.ResNetBlock(512,512),
    yy.myNN.ResNetBlock(512,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,256),
    yy.myNN.ResNetBlock(256,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,128),
    yy.myNN.ResNetBlock(128,64),
    yy.myNN.ResNetBlock(64,64),
    nn.ConvTranspose2d(64,64,7,groups=64,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
    yy.myNN.ResNetBlock(64,64),
    nn.ConvTranspose2d(64,64,3),nn.BatchNorm2d(64),nn.ReLU(),
    yy.myNN.ResNetBlock(64,64),
    yy.myNN.ResNetBlock(64,32),
    yy.myNN.ResNetBlock(32,32),
    yy.myNN.ResNetBlock(32,32),
    yy.myNN.ResNetBlock(32,20),
    yy.myNN.ResNetBlock(20,20),
    yy.myNN.ResNetBlock(20,20),
    CoordRegressionNetwork(20),
)'''
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
    yy.myNN.ResNetBlock(512,512),#结束resnet
    nn.Conv2d(in_channels=512,out_channels=512,groups=512,kernel_size=3,padding=1,stride=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=512,dilation=1,groups=1,kernel_size=1,padding=0,stride=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2,stride=2),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,groups=512,padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1),
    nn.BatchNorm2d(1024),
    nn.ReLU(),###开始分叉
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,groups=1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,groups=1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=20,kernel_size=1),
    nn.BatchNorm2d(20),
    nn.ReLU(),
    CoordRegressionNetwork(20)
)
# 测试
testx=torch.randn(size=(1,3,400,400),device='cuda',dtype=torch.float32)
net.to('cuda')
print('测试网络输入',testx.shape)
with torch.no_grad():
    print('测试网络输出',net(testx)[0].shape)
# torch.onnx.export(net,torch.randn(size=(1,3,300,400)).to('cuda'),outputONNXPath)
# plt.show()
# exit()

#训练数据
allVideo, allJoints=yy.dataProcess.loadYYDataToNumpy(fileName=fileName)
#归一化
allJoints=yy.dataProcess.JointsNormalization(allJonits=allJoints)
amass52to20,amass20pt=yy.MoViConst.getAmass20()
allJoints=allJoints[:,amass52to20,:]#转为amass20
##提取二维坐标
allJoints=allJoints[...,[0,1]]
##对视频填充为正方形
ca_rand=np.random.randint(size=(len(allVideo),3,50,400),high=256,low=0,dtype='uint8')
allVideo=np.concatenate((ca_rand,allVideo,ca_rand),axis=2)

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
#归一化
tj=yy.dataProcess.JointsNormalization(allJonits=tj)
tj=tj[:,amass52to20,:]#转为amass20
#提取二维坐标
tj=tj[...,[0,1]]
#视频填充为正方形
ca_rand=np.random.randint(size=(len(tv),3,50,400),high=256,low=0,dtype='uint8')
tv=np.concatenate((ca_rand,tv,ca_rand),axis=2)

tv=torch.from_numpy(tv)
tj=torch.from_numpy(tj)
evalData=yy.myNN.tensorDataset(tv,tj)
evalBatchSize=5
evalData=DataLoader(evalData,batch_size=evalBatchSize,shuffle=True)
evalData=cycle(evalData)

def mseAndCosineLoss(a,b,propotion=0.0001):
    mse=F.mse_loss(a,b)
    co=torch.sum(1-torch.cosine_similarity(a, b,dim=2))
    return torch.add(torch.mul(co,propotion),torch.mul(mse,1-propotion))


#正式训练
net.apply(yy.myNN.init_weights)
net.to('cuda')
# optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
# optimizer = torch.optim.RMSprop(net.parameters(), lr=learningRate)
# loss=F.mse_loss
# loss=F.l1_loss
# loss=F.pairwise_distance
loss= mseAndCosineLoss

num_batches =  len(trainData)
timer=yy.Timer()

for epoch in range(epochSize):
    net.train()
    for i, (X, y) in enumerate(trainData):
        X=(X/255).float()
        timer.start()
        X, y = X.to('cuda'), y.to('cuda')
        y_hat, heatmaps = net(X)
        euc_losses = dsntnn.euclidean_losses(y_hat, y)
        reg_losses = dsntnn.js_reg_losses(heatmaps, y, sigma_t=sigma_t)
        l = dsntnn.average_loss(euc_losses + reg_losses)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        yl=l.cpu().detach().numpy().tolist()#用于输出训练集的loss
        ##评估
        ex,ey=next(evalData)
        ex=(ex/255).float()
        ex,ey=ex.to('cuda'),ey.to('cuda')
        with torch.no_grad():
            tyHat,tmaps=net(ex)
            tl=loss(tyHat, ey)
            tl=tl.cpu().detach().numpy().tolist()*10
        picture.draw([epoch+i/num_batches,yl],[epoch+i/num_batches,tl])
        timer.stop()
    print('第',epoch+1,'个epoch训练完成')
print('共需要时间',timer.sum(),'秒')

# with torch.no_grad():
#     tx,ty=next(iter(trainData))
#     tx,ty=tx.to('cuda'),ty.to('cuda')
#     tx=(tx/255).float()
#     ex,ey=next(evalData)
#     ex,ey=ex.to('cuda'),ey.to('cuda')
#     ex=(ex/255).float()
#     tyH=net(tx)
#     eyH=net(ex)
#     print('训练集误差',loss(ty, tyH))
#     print('评估集误差',loss(ey, eyH))
# net.to('cpu')
torch.onnx.export(net,torch.randn(size=(1,3,400,400)).to('cuda'),outputONNXPath)
plt.show()
