
import torch
import yy 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
from torch.nn import functional as F
import matplotlib.pyplot as plt
from itertools import cycle
import yydsntnn as dsntnn
import keyboard
'''
使用448movi
相比于v2   加入dsntnn   并使用热图进行训练  
'''
# yy.tools.setup_seed(0)
outputPath='C:/Users/yuanying/Desktop'
dataPath='D:/1Datasets/MoVi/yyData'
# dataPath='D:/1Datasets/Human3.6M/yyData'
datasetName=1 # 1为movi 2为HM36
outputFormat=2 # 1为onnx 2为torchNet参数 3为torchNet整个网络
LoadParameter=0 # 0 为不加载预训练模型 1 为加载预训练模型

epochSize = 100
batchSize = 2
learningRate = 0.1e-4
sigma_t=1.2


if datasetName==1: 
    fileName='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_3.npy'
    fileNametest='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_3.npy'
elif datasetName==2:
    fileName='HM36_[5, 6, 7, 8, 9]_Inte_50.npy'
    # fileName='HM36_[1]_Inte_25.npy'
    fileNametest='HM36_[1]_Inte_25.npy'

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
            yy.myNN.ResNetBlock(64,64),
            yy.myNN.ResNetBlock(64,128),
            yy.myNN.ResNetBlock(128,128),
            yy.myNN.ResNetBlock(128,128),
            yy.myNN.ResNetBlock(128,256),
            yy.myNN.ResNetBlock(256,256),
            yy.myNN.ResNetBlock(256,256),
            yy.myNN.ResNetBlock(256,256),
            yy.myNN.ResNetBlock(256,256),
            nn.Conv2d(in_channels=256,out_channels=1,kernel_size=3,padding=1,bias=False)
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
if datasetName==1:
    jointNum=20
if datasetName==2:
    jointNum=17
block_type, layers, _channels, _name = yy.myNN.getResnetBackboneSpec(101)
net=nn.Sequential(
    yy.myNN.ResNetBackbone(block_type, layers,in_channel=3),
    nn.Conv2d(in_channels=2048,out_channels=512,groups=512,kernel_size=3,padding=1,stride=1,bias=False),
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
    nn.Conv2d(in_channels=1024,out_channels=28*jointNum,kernel_size=1,bias=False),
    nn.BatchNorm2d(28*jointNum),
    nn.ReLU(),
    dim3dsntnn(channels=28*jointNum,jointNum=jointNum)
)
# 测试
testx=torch.randn(size=(2,3,448,448),device='cuda',dtype=torch.float32)
net.to('cuda')
print('测试网络输入',testx.shape)
with torch.no_grad():
    print('测试网络输出',net(testx)[0].shape,net(testx)[1].shape,net(testx)[2].shape,net(testx)[3].shape)
    # print('测试网络输出',net(testx).shape)
# torch.onnx.export(net,torch.randn(size=(1,3,300,400)).to('cuda'),outputONNXPath)
# plt.show()

#训练数据
allVideo, allJoints=yy.dataProcess.loadYYDataToNumpy(fileName=fileName,dataPath=dataPath)
#归一化
allJoints,NormScale,NormBias=yy.dataProcess.JointsNormalization(allJonits=allJoints,mode=1)
print('训练集归一化参数',NormScale,NormBias)##注意此时没有转坐标系
with open(outputPath+'/NormScale_NormBias.npy', 'wb') as f:
    np.save(f, NormScale)
    np.save(f, NormBias)
if datasetName==1:
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
tv,tj=yy.dataProcess.loadYYDataToNumpy(fileName=fileNametest,dataPath=dataPath)
#归一化
# tj=(tj+NormBias)/NormScale #由于后面要计算MGJPE所以先不归一化

if datasetName==1:
    tj=tj[:,amass52to20,:]#转为amass20

tv=torch.from_numpy(tv)
tj=torch.from_numpy(tj)
evalData=yy.myNN.tensorDataset(tv,tj)
evalBatchSize=2
evalData=DataLoader(evalData,batch_size=evalBatchSize,shuffle=True)
evalData=cycle(evalData)

def mseAndCosineLoss(a,b,propotion=0.):
    mse=F.mse_loss(a,b)
    co=torch.sum(1-torch.cosine_similarity(a, b,dim=2))
    return torch.add(torch.mul(co,propotion),torch.mul(mse,1-propotion))

NormScale=torch.from_numpy(NormScale).to('cuda')
NormBias=torch.from_numpy(NormBias).to('cuda')
#正式训练
if LoadParameter==0:
    net.apply(yy.myNN.init_weights)
    print('成功初始化网络')
elif LoadParameter==1:
    net.load_state_dict(torch.load(outputPath+'/outputParameter.pt'))
    print('成功加载预训练模型')
net.to('cuda')
# optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
# optimizer = torch.optim.RMSprop(net.parameters(), lr=learningRate)

num_batches =  len(trainData)
timer=yy.Timer()
picture=yy.drawSomeLines(2,xlim=[0,epochSize],ylim=[0,1],interval=int(num_batches/100))
stop=False
for epoch in range(epochSize):
    net.train()
    for i, (X, y) in enumerate(trainData):
        X=(X/255).float()
        timer.start()
        X, y = X.to('cuda'), y.to('cuda')
        xyzHat,xymap,zxmap,zymap = net(X)
        euc_losses = dsntnn.euclidean_losses(xyzHat, y)
        xymap_reg_losses = dsntnn.js_reg_losses(xymap, y[...,[0,1]], sigma_t=sigma_t)
        zxmap_reg_losses = dsntnn.js_reg_losses(zxmap, y[...,[2,0]], sigma_t=sigma_t)
        zymap_reg_losses = dsntnn.js_reg_losses(zymap, y[...,[2,1]], sigma_t=sigma_t)
        l = dsntnn.average_loss(xymap_reg_losses + zxmap_reg_losses + zymap_reg_losses + euc_losses*2)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        yl=l.cpu().detach().numpy().tolist()#用于输出训练集的loss
        ##评估

        ex,ey=next(evalData)
        ex=(ex/255).float()
        ex,ey=ex.to('cuda'), ey.to('cuda')
        with torch.no_grad():
            e_xyzHat,e_xymap,e_zxmap,e_zymap=net(ex)
            tl=yy.dataProcess.MPJPE(e_xyzHat*NormScale-NormBias, ey)
            tl=tl.cpu().detach().numpy().tolist()
        # tl=0.
        # picture.draw([epoch+i/num_batches,yl],[epoch+i/num_batches,tl/1000])
        timerBatch=timer.stop()
        print('第',epoch+1,'/',epochSize,'个epoch中',i,'/',num_batches,'个batch训练完成','用时',format(timerBatch,'.2f'),'MPJPE:',format(tl,'.2f'))
        if keyboard.is_pressed('esc') and keyboard.is_pressed('p'):
            stop=True
            break
    if stop:
        break
    print('第',epoch+1,'个epoch训练完成','此时trainLoss:',yl,'此时evalLoss:',tl,'已用时间:',timer.sum())
print('共需要时间',timer.sum(),'秒')
timer.start()
print('正在导出模型(参数为',outputFormat,')......')
net.eval()
X, y = X.to('cpu'), y.to('cpu')
ex,ey=ex.to('cpu'), ey.to('cpu')

if outputFormat==1:
    torch.onnx.export(net,torch.randn(size=(1,3,448,448)).to('cuda'),outputPath+'/outputONNX.onnx')
elif outputFormat==2:
    torch.save(net.state_dict(),outputPath+'/outputParameter.pt')
print('导出模型用时:',timer.stop(),'模型参数为：',outputFormat)
print('----训练完成并成功导出模型----请等待程序结束')
# plt.pause(0)
