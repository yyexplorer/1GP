
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
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode
'''
使用448movi
movi 此网络在投影分支中权值不共享 mpjpe为20
hm 6789 mpjpe 168.9
'''
# yy.tools.setup_seed(0)
outputPath='C:/Users/yuanying/Desktop'
datasetName=1 # 1为movi 2为HM36
outputFormat=1 # 1为onnx 2为torchNet参数 3为torchNet整个网络
LoadParameter=0 # 0 为不加载预训练模型 1 为加载预训练模型

epochSize = 15
batchSize = 4
learningRate = 0.03e-4
sigma_t=1.3

if datasetName==1:
    dataPath='D:/1Datasets/MoVi/yyData'
elif datasetName==2:
    dataPath='D:/1Datasets/Human3.6M/yyData'
if datasetName==1: 
    fileName='MoViData_Subj_46_47_50_55_60_65_70_71_75_80_81_82_83_84_85_88_89_90_Re_Tr_Ct_Inte_10.npy'
    fileName='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_2.npy'
    fileNametest='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_3.npy'
elif datasetName==2:
    fileName='HM36_[1]_Inte_25.npy'
    fileNametest='HM36_[1]_Inte_199.npy'

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
        self.resnet1=nn.Sequential(
            nn.Conv2d(1,128,kernel_size=5,padding=2),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,1,kernel_size=3,padding=1),nn.ReLU(),
            )
        self.resnet2=nn.Sequential(
            nn.Conv2d(1,128,kernel_size=5,padding=2),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,1,kernel_size=3,padding=1),nn.ReLU(),
            )    
        self.resnet3=nn.Sequential(
            nn.Conv2d(1,128,kernel_size=5,padding=2),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,1,kernel_size=3,padding=1),nn.ReLU(),
            )    
    def forward(self,x):
        # print('dsntnn输入',x.shape)#torch.Size([2, 476, 28, 28])
        xymap=[]
        zxmap=[]
        zymap=[]
        heatmapAll=[]
        for i in range(self.jointNum):
            '''
            对每一个节点对应的三维heatmap进行操作
            将单个节点的三维heatmap利用dsnt计算xyz坐标
            输入tensor：batch*channels*x*y
            输出tensor：batch*1*3, xymap, zxmap, zymap
            '''
            heatmapi=x[:,i*self.interval:(i+1)*self.interval,:,:]
            heatmapAll.append(heatmapi.unsqueeze(1))
        heatmapAll=torch.concat(heatmapAll,dim=1)
        # print('heatmapAll前',heatmapAll.shape)#torch.Size([2,17, 28, 28, 28])
        heatmapAll=heatmapAll.view(heatmapAll.shape[0]*heatmapAll.shape[1],heatmapAll.shape[2],heatmapAll.shape[3],heatmapAll.shape[4])
        # print('heatmapAll后',heatmapAll.shape)#torch.Size([2*17, 28, 28, 28])
        unnormalized_xymap=self.resnet1(torch.max(heatmapAll,dim=1)[0].unsqueeze(1))
        unnormalized_zxmap=self.resnet1(torch.max(heatmapAll,dim=3)[0].unsqueeze(1))
        unnormalized_zymap=self.resnet1(torch.max(heatmapAll,dim=2)[0].unsqueeze(1))
        # print('unnormalized_xymap',unnormalized_xymap.shape)#torch.Size([34, 1, 28, 28])
        xymap = (dsntnn.d2_softmax(unnormalized_xymap)).view(-1,jointNum,unnormalized_xymap.shape[2],unnormalized_xymap.shape[3])
        zxmap = (dsntnn.d2_softmax(unnormalized_zxmap)).view(-1,jointNum,unnormalized_zxmap.shape[2],unnormalized_zxmap.shape[3])
        zymap = (dsntnn.d2_softmax(unnormalized_zymap)).view(-1,jointNum,unnormalized_zymap.shape[2],unnormalized_zymap.shape[3])
        # print('xymap',xymap.shape)#torch.Size([2, 17, 28, 28])
        #shape  batch*jointNum*3
        xy=dsntnn.dsnt(xymap)
        zx=dsntnn.dsnt(zxmap)
        zy=dsntnn.dsnt(zymap)
        # print('xy',xy.shape)#torch.Size([2, 17, 2])
        xyz=torch.concat([
            ((xy[...,0]+zx[...,1])/2).unsqueeze(2),
            ((xy[...,1]+zy[...,1])/2).unsqueeze(2),
            ((zx[...,0]+zy[...,0])/2).unsqueeze(2)
            ],dim=2)
        # print('xyz',xyz.shape)#torch.Size([2, 17, 3])
        return xyz,xymap,zxmap,zymap
if datasetName==1:
    jointNum=20
if datasetName==2:
    jointNum=17
block_type, layers, _channels, _name = yy.myNN.getResnetBackboneSpec(50)
net=nn.Sequential(
    yy.myNN.ResNetBackbone(block_type, layers,in_channel=3),
    # yy.myNN.ResNetBlock(2048,512),
    nn.Conv2d(in_channels=2048,out_channels=2048,groups=2048,kernel_size=3,padding=1,stride=1,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=2048,dilation=1,groups=1,kernel_size=1,padding=0,stride=1,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=2048,out_channels=2048,kernel_size=2,stride=2,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,groups=2048,padding=1,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),###开始分叉
    nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,padding=1,groups=2048,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,padding=1,groups=2048,bias=False),
    nn.BatchNorm2d(2048),
    nn.ReLU(),
    nn.Conv2d(in_channels=2048,out_channels=28*jointNum,kernel_size=1,bias=False),
    nn.BatchNorm2d(28*jointNum),
    nn.ReLU(),
    dim3dsntnn(channels=28*jointNum,jointNum=jointNum)
)
# 测试
testx=torch.randn(size=(batchSize,3,448,448),device='cuda',dtype=torch.float32)
net.to('cuda')
print('测试网络输入',testx.shape)
with torch.no_grad():
    _testy=net(testx)
    print('测试网络输出',_testy[0].shape,_testy[1].shape,_testy[2].shape,_testy[3].shape)
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
evalBatchSize=1
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
forwardAvgTim=yy.Timer()
picture=yy.drawSomeLines(2,xlim=[0,epochSize],ylim=[0,400],interval=int(num_batches/20),mode=2)
stop=False
for epoch in range(epochSize):
    if epoch < 5:
        yy.myNN.changeLr(optimizer,learningRate*40)
    elif epoch < 10:
        yy.myNN.changeLr(optimizer,learningRate*15)
    else:
        yy.myNN.changeLr(optimizer,learningRate)
    net.train()
    for i, (X, y) in enumerate(trainData):
        ##训练
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
            forwardAvgTim.start()
            e_xyzHat,e_xymap,e_zxmap,e_zymap=net(ex)
            forwardAvgTim.stop()
            tl=yy.dataProcess.MPJPE(e_xyzHat*NormScale-NormBias, ey)
            tl=tl.cpu().detach().numpy().tolist()
            ylMPJPE=yy.dataProcess.MPJPE(xyzHat*NormScale-NormBias, y*NormScale-NormBias)
            ylMPJPE=ylMPJPE.cpu().detach().numpy().tolist()
            # yy.tools.showMotionAndVideo(yy.tools.changeCoord((e_xyzHat*NormScale-NormBias).cpu().detach().numpy()),yy.dataProcess.HM36Const.getParent(),1,ex.cpu().detach().numpy(),azim=-70,elev=-170)
        picture.draw([epoch+i/num_batches,ylMPJPE],[epoch+i/num_batches,tl])
        timerBatch=timer.stop()
        print('第',epoch+1,'/',epochSize,'个epoch中',i,'/',num_batches,'个batch完成','用时',format(timerBatch,'.1f'),'EvalMPJPE:',format(tl,'.1f'),'TrainMPJPE:',format(ylMPJPE,'.1f'))
        if keyboard.is_pressed('esc') and keyboard.is_pressed('p'):
            stop=True
            break
    if stop:
        break
    print('第',epoch+1,'个epoch训练完成','此时trainLoss:',yl,'此时evalLoss:',tl,'已用时间:',timer.sum())
print('共需要时间',timer.sum(),'秒')
print('模型正向运行平均时间:',forwardAvgTim.avg())
timer.start()
print('正在导出模型(参数为',outputFormat,')......')
X, y = X.to('cpu'), y.to('cpu')
ex,ey=ex.to('cpu'), ey.to('cpu')
if outputFormat==1:
    torch.onnx.export(net,torch.randn(size=(1,3,448,448)).to('cuda'),outputPath+'/outputONNX.onnx',training=TrainingMode.PRESERVE,do_constant_folding=False,opset_version=12)
elif outputFormat==2:
    torch.save(net.state_dict(),outputPath+'/outputParameter.pt')
print('导出模型用时:',timer.stop(),'模型参数为：',outputFormat)
print('----训练完成并成功导出模型----请等待程序结束')
plt.pause(0)
