
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
使用448movi
相比于v2   加入dsntnn   并使用热图进行训练  
'''
yy.tools.setup_seed(0)
outputONNXPath='C:/Users/yuanying/Desktop/testonnx.onnx'
dataPath='D:/1Datasets/MoVi'
# fileName='MoViData_Subj_86_87_Re_Tr_Inte_10.npy'
fileName='MoViData_Subj_46_47_50_55_60_65_70_71_75_80_81_82_83_84_85_88_89_90_Re_Tr_Ct_Inte_10.npy'
epochSize = 5
batchSize = 10
learningRate = 1e-4
sigma_t=1.2
picture=yy.drawSomeLines(2,xlim=[0,epochSize],ylim=[0,1])

#网络
class dim3dsntnn(nn.Module):
    '''
    参数：节点数
    默认20个节点
    输入维度为：batch*channels*x*y
    默认x等于y
    channels
    '''
    def __init__(self,channels,jointNum=20):
        super().__init__()
        assert channels % jointNum==0
        self.channels=channels
        self.jointNum=jointNum
        self.interval=int(self.channels/self.jointNum)
        self.output=[]
        self.resnet=nn.Sequential(
            yy.myNN.ResNetBlock(1,64),
            yy.myNN.ResNetBlock(64,64),
            yy.myNN.ResNetBlock(64,128),
            yy.myNN.ResNetBlock(128,64),
            yy.myNN.ResNetBlock(64,64),
            yy.myNN.ResNetBlock(64,1),
            )##

    def forward(self,x):
        xymap=[]
        zxmap=[]
        zymap=[]
        for i in range(self.jointNum):
            '''
            对每一个节点对应的三维heatmap进行操作
            将单个节点的三维heatmap利用dsnt计算xyz坐标
            输入tensor：batch*channels*x*y
            输出tensor：batch*1*3, xymap, zxmap, zymap
            '''
            heatmapi=x[:,i*self.interval:(i+1)*self.interval,:,:]
            #xy 选择max还是mean
            # unnormalized_xymap=self.resnet(torch.mean(heatmapi,dim=1).unsqueeze(1))
            unnormalized_xymap=self.resnet(torch.max(heatmapi,dim=1)[0].unsqueeze(1))
            # print(unnormalized_xymap.shape)#torch.Size([2, 1, 26, 26])
            xymapi = dsntnn.d2_softmax(unnormalized_xymap)
            xymap.append(xymapi)
            #zx选择max还是mean
            # unnormalized_zxmap=self.resnet(torch.mean(heatmapi,dim=3).unsqueeze(1))
            unnormalized_zxmap=self.resnet(torch.max(heatmapi,dim=3)[0].unsqueeze(1))
            zxmapi = dsntnn.d2_softmax(unnormalized_zxmap)
            zxmap.append(zxmapi)
            #zy选择max还是mean
            # unnormalized_zymap=self.resnet(torch.mean(heatmapi,dim=2).unsqueeze(1))
            unnormalized_zymap=self.resnet(torch.max(heatmapi,dim=2)[0].unsqueeze(1))
            zymapi=dsntnn.d2_softmax(unnormalized_zymap)
            zymap.append(zymapi)
        #heatmap的shape batch*jointNum*x*y
        xymap=torch.concat(xymap,dim=1)
        zxmap=torch.concat(zxmap,dim=1)
        zymap=torch.concat(zymap,dim=1)
        #shape  batch*jointNum*3
        xy=dsntnn.dsnt(xymap)
        zx=dsntnn.dsnt(zxmap)
        zy=dsntnn.dsnt(zymap)
        xyz=torch.concat([
            ((xy[...,0]+zx[...,1])/2).unsqueeze(2),
            ((xy[...,1]+zy[...,1])/2).unsqueeze(2),
            ((zx[...,0]+zy[...,0])/2).unsqueeze(2)
            ],dim=2)
        # print('xyz',xyz.shape)
        return xyz,xymap,zxmap,zymap
net=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2,bias=False),
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
    nn.Conv2d(in_channels=512,out_channels=512,groups=512,kernel_size=3,padding=1,stride=1,bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=512,dilation=1,groups=1,kernel_size=1,padding=0,stride=1,bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,groups=512,padding=1,bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(),###开始分叉
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,groups=1024,bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,groups=1024,bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024,out_channels=28*20,kernel_size=1,bias=False),
    nn.BatchNorm2d(28*20),
    nn.ReLU(),
    dim3dsntnn(channels=28*20,jointNum=20)
)
# 测试
testx=torch.randn(size=(2,3,448,448),device='cuda',dtype=torch.float32)
net.to('cuda')
print('测试网络输入',testx.shape)
with torch.no_grad():
    print('测试网络输出',net(testx)[0].shape,net(testx)[1].shape,net(testx)[2].shape,net(testx)[3].shape)
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
# allJoints=allJoints[...,[0,1]]
##对视频填充为正方形
# ca_rand=np.random.randint(size=(len(allVideo),3,50,400),high=256,low=0,dtype='uint8')
# allVideo=np.concatenate((ca_rand,allVideo,ca_rand),axis=2)

allVideo=torch.from_numpy(allVideo)
allJoints=torch.from_numpy(allJoints)
print('allVideo',allVideo.shape,allVideo.dtype)
print('allJoints',allJoints.shape,allJoints.dtype)
trainData=yy.myNN.tensorDataset(allVideo,allJoints)
trainData=DataLoader(trainData,batch_size=batchSize,shuffle=True)
print('训练数据 迭代器',type(trainData),len(trainData))
#评估数据
fileNametest='MoViData_Subj_86_87_Re_Tr_Ct_Inte_10.npy'
tv,tj=yy.dataProcess.loadYYDataToNumpy(fileName=fileNametest)
#归一化
tj=yy.dataProcess.JointsNormalization(allJonits=tj)
tj=tj[:,amass52to20,:]#转为amass20
#提取二维坐标
# tj=tj[...,[0,1]]
#视频填充为正方形
# ca_rand=np.random.randint(size=(len(tv),3,50,400),high=256,low=0,dtype='uint8')
# tv=np.concatenate((ca_rand,tv,ca_rand),axis=2)

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
        xyzHat,xymap,zxmap,zymap = net(X)
        # print(xyzHat.shape)
        # print(y.shape)
        # exit()
        euc_losses = dsntnn.euclidean_losses(xyzHat, y)
        # reg_losses = dsntnn.js_reg_losses(heatmaps, y, sigma_t=sigma_t)
        xymap_reg_losses = dsntnn.js_reg_losses(xymap, y[...,[0,1]], sigma_t=sigma_t)
        zxmap_reg_losses = dsntnn.js_reg_losses(zxmap, y[...,[2,0]], sigma_t=sigma_t)
        zymap_reg_losses = dsntnn.js_reg_losses(zymap, y[...,[2,1]], sigma_t=sigma_t)
        l = dsntnn.average_loss(xymap_reg_losses + zxmap_reg_losses + zymap_reg_losses + euc_losses)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        yl=l.cpu().detach().numpy().tolist()#用于输出训练集的loss
        ##评估
        '''
        ex,ey=next(evalData)
        ex=(ex/255).float()
        ex,ey=ex.to('cuda'), ey.to('cuda')
        with torch.no_grad():
            e_xyzHat,e_xymap,e_zxmap,e_zymap=net(ex)
            tl=loss(e_xyzHat, ey)
            tl=tl.cpu().detach().numpy().tolist()*10
        '''
        picture.draw([epoch+i/num_batches,yl],[epoch+i/num_batches,0])
        timer.stop()
    print('第',epoch+1,'个epoch训练完成','此时trainLoss:',yl,'此时evalLoss:',0)
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
torch.onnx.export(net,torch.randn(size=(1,3,448,448)).to('cuda'),outputONNXPath)
plt.pause(0)
