
import re
from turtle import forward
from IPython import display
import cv2
import numpy as np
import MoViutils
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import time
import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import yydsntnn as dsntnn
import json
from torchvision.models.resnet import BasicBlock, Bottleneck

class tools:

    def setup_seed(seed):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def getFrameFromVideo(videoPath,frameNum,mode=0):

        cap = cv2.VideoCapture(videoPath)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameNum-1)
        success , frame = cap.read()
        if not success:
            print("there are some errors in getFrameFromVideo()")
            return
        if mode == 0:
            cv2.imshow('frame',frame)
            cv2.waitKey(0)
            cap.release()
        if mode == 1:
            out=np.array(frame)
            out=out.astype(np.float32)/255
            out=out[...,[2,1,0]]
            out=out.swapaxes(0,2).swapaxes(1,2)
            return out 
        if mode == 2:
            return frame
        if mode == 3:
            address=videoPath+'_frameNum_'+str(frameNum)+'.jpg'
            cv2.imwrite(address,frame)
            return

    def showTreeOfDict(ndict, indent=0, print_type=False):

        if isinstance(ndict, dict):
            for key, value in ndict.items():
                if print_type:
                    print(
                        "\t" * indent + "Key: " + str(key) + ",\t" + "Type: ", type(value),
                    )
                    tools.showTreeOfDict(value, indent + 1, True)
                else:
                    print("\t" * indent + "Key: " + str(key))
                    tools.showTreeOfDict(value, indent + 1)

    def mat2dict(filePath):

        # Reading MATLAB file
        data = sio.loadmat(filePath, struct_as_record=False, squeeze_me=True)

        # Converting mat-objects to a dictionary
        for key in data:
            if key != "__header__" and key != "__global__" and key != "__version__":
                if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
                    data_out = MoViutils.matobj2dict(data[key])
        return data_out

    def showMotion(markerlocation,parent,fps,save=0,mark=None):
 
        if parent is not None:
            assert markerlocation.shape[1]==parent.shape[0]
        #画图基本设置
        fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        #ax = plt.gca(projection="3d")
        ax=plt.subplot(projection="3d")
        # ax.set(xlim3d=(np.min(markerlocation[...,0]), 1000), xlabel='X')
        # ax.set(ylim3d=(-1000, 1000), ylabel='Y')
        # ax.set(zlim3d=(0, 2000), zlabel='Z')


        maxx,minx=np.max(markerlocation[...,0]),np.min(markerlocation[...,0])
        maxy,miny=np.max(markerlocation[...,1]),np.min(markerlocation[...,1])
        maxz,minz=np.max(markerlocation[...,2]),np.min(markerlocation[...,2])
        side=np.max([maxx-minx,maxy-miny,maxz-minz])
        ax.set(xlim3d=(maxx-side,maxx), xlabel='X')
        ax.set(ylim3d=(maxy-side,maxy), ylabel='Y')
        ax.set(zlim3d=(maxz-side,maxz), zlabel='Z')
        #print(ax.azim)
        ax.azim=-90
        #print(ax.elev)
        ax.elev=-90
        #初始化各项参数
        frameNum=markerlocation.shape[0]#帧总数
        pointNum=markerlocation.shape[1]#每一帧所用的点的数量
        lineNum=1#每一帧需要画的线的数量
        #初始化线（一条线）
        if parent is not None:
            lines = [ax.plot([], [], [],color='g')[0]for _ in parent]
        #初始化点（一个）
        # points= [ax.plot([],[],[],color='r',marker='o',markersize=2)[0] for _ in range(pointNum)]
        points=[]
        for i in range(pointNum):
            if i==mark:
                points.append(ax.plot([],[],[],color='b',marker='o',markersize=4)[0])
            else:
                points.append(ax.plot([],[],[],color='r',marker='o',markersize=2)[0])

        # ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
        def drawLines(num):
            for line,j,p in zip(lines,range(parent.shape[0]),parent):
                if p == -1:
                    continue
                a=np.array([markerlocation[num,j,0],markerlocation[num,p,0]])
                b=np.array([markerlocation[num,j,1],markerlocation[num,p,1]])
                c=np.array([markerlocation[num,j,2],markerlocation[num,p,2]])
                line.set_data(a,b)
                line.set_3d_properties(c)
            return
        def drawDots(num):
            for point,i in zip(points,range(pointNum)):
                point.set_data([markerlocation[num,i,0]],[markerlocation[num,i,1]])
                point.set_3d_properties([markerlocation[num,i,2]])
            return

        #画图
        def update(num):
            # NOTE: there is no .set_data() for 3 dim data...
            drawDots(num)
            if parent is not None:
                drawLines(num)
            return


        ani=animation.FuncAnimation(fig=fig, 
                                    func=update,
                                    frames=range(frameNum),  
                                    interval=1000/fps,
                                    repeat=True)
        if save==1:
            ani.save('1.gif')
        plt.show()
        return

    def showMotionAndVideo(markerlocation,parent,fps,video=None,videoBegin=0,save=0,show=1,normlization=0,azim=1,elev=1):

        if parent is not None:
            assert markerlocation.shape[1]==parent.shape[0]
        #画图基本设置
        fig = plt.figure()
        if video is None:
            # ax = fig.add_subplot(projection="3d")
            #ax = plt.gca(projection="3d")
            ax=plt.subplot(projection="3d")
        else:
            gs = gridspec.GridSpec(1, 2)
            ax =plt.subplot(gs[0,1],projection="3d")#右侧显示肢体轨迹
            ax2=plt.subplot(gs[0,0])
  
        # lmax=np.max(markerlocation)
        # lmin=np.min(markerlocation)
        if normlization==0:
        #     ax.set(xlim3d=(lmin, lmax), xlabel='X')
        #     ax.set(ylim3d=(lmin, lmax), ylabel='Y')
        #     ax.set(zlim3d=(lmin, lmax), zlabel='Z')
            maxx,minx=np.max(markerlocation[...,0]),np.min(markerlocation[...,0])
            maxy,miny=np.max(markerlocation[...,1]),np.min(markerlocation[...,1])
            maxz,minz=np.max(markerlocation[...,2]),np.min(markerlocation[...,2])
            side=np.max([maxx-minx,maxy-miny,maxz-minz])
            ax.set(xlim3d=(maxx-side,maxx), xlabel='X')
            ax.set(ylim3d=(maxy-side,maxy), ylabel='Y')
            ax.set(zlim3d=(maxz-side,maxz), zlabel='Z')
        else:
            ax.set(xlim3d=(-1, 1), xlabel='X')
            ax.set(ylim3d=(-1, 1), ylabel='Y')
            ax.set(zlim3d=(0, 1), zlabel='Z')
        #print(ax.azim)
        ax.azim=azim
        #print(ax.elev)
        ax.elev=elev
        #初始化各项参数
        frameNum=markerlocation.shape[0]#帧总数
        pointNum=markerlocation.shape[1]#每一帧所用的点的数量
        lineNum=1#每一帧需要画的线的数量
        #初始化线（一条线）
        lines = [ax.plot([],[],[],color='g')[0]for _ in parent]
        #初始化点（一个）
        points= [ax.plot([],[],[],color='r',marker='o',markersize=2)[0] for _ in range(pointNum)]
        # ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
        def drawLines(num):
            for line,j,p in zip(lines,range(parent.shape[0]),parent):
                if p == -1:
                    continue
                a=np.array([markerlocation[num,j,0],markerlocation[num,p,0]])
                b=np.array([markerlocation[num,j,1],markerlocation[num,p,1]])
                c=np.array([markerlocation[num,j,2],markerlocation[num,p,2]])
                line.set_data(a,b)
                line.set_3d_properties(c)
            return
        def drawDots(num):
            for point,i in zip(points,range(pointNum)):
                point.set_data([markerlocation[num,i,0]],[markerlocation[num,i,1]])
                point.set_3d_properties([markerlocation[num,i,2]])
            return
        def showVideoByCap(num):
            video.set(cv2.CAP_PROP_POS_FRAMES,num+videoBegin)
            success , img = video.read()
            if success:
                img = img[..., ::-1]
                ax2.imshow(img)
            return
        def showVideoByNumpy(num):
            img = video[num,...].swapaxes(1,2).swapaxes(0,2)
            ax2.imshow(img)
            return
        #画图
        def update(num):
            # NOTE: there is no .set_data() for 3 dim data...
            drawDots(num)
            drawLines(num)
            if video is not None:
                if type(video)==np.ndarray:
                    showVideoByNumpy(num)
                else:
                    showVideoByCap(num)
            return
        if frameNum==1:
            update(0)
        else:
            ani=animation.FuncAnimation(fig=fig, 
                                    func=update,
                                    frames=range(frameNum),  
                                    interval=1000/fps,
                                    repeat=True)
        
        if save==1:
            ani.save('output.gif')
        if show==1:
            plt.show()
        return
    
    def showImage(mat,mode=0):

        if mode==0:
            plt.imshow(mat.swapaxes(1,2).swapaxes(0,2))
            plt.show()
        return
    def showImageAndAction(image,markerlocation,parent):

        fig,ax = plt.subplots()
        pointNum=markerlocation.shape[0]
        image=image.swapaxes(1,2).swapaxes(0,2)
        ax.imshow(image)

        def drawDots():
            for i in range(pointNum):
                plt.plot(markerlocation[i,0],markerlocation[i,1],color='r',marker='o',markersize=2)
            return
        # drawLines()
        drawDots()
        plt.show()
    def changeCoord(mat,mode=0):

        if mode==0:
            mat=mat[...,[0,2,1]]
            mat[...,1]=-mat[...,1]
        return mat
class MoViConst:

    def parentAmass52():
        '''
        原版amass52数据
        '''
        parent=np.array([ 0,  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 10, 13,
                        14, 15, 17, 18, 19, 20, 21, 23, 24, 21, 26, 27, 21, 29, 30, 21,
                        32, 33, 21, 35, 36, 22, 38, 39, 22, 41, 42, 22, 44, 45, 22, 47,
                        48, 22, 50, 51])-1
        return parent
    def getAmass20():
        '''
        从amass52提取20个关键点
        返回提取矩阵，amass20父节点
        amass数据标记
        0 肚子6
        1 左臀0
        2 右臀0
        4 左膝盖1
        5 右膝盖2
        6 胸腔-1
        7 左脚后跟4
        8 右脚后跟5
        10 左脚脚趾7
        11 右脚脚趾8
        12 脖子上部6
        15 鼻子12
        16 左肩6
        17 右肩6
        18 左肘部16
        19 右肘部17
        20 左手腕18
        21 右手腕19
        27 左手中指20
        42 右手中指21
        '''
        amass20=[0,1,2,4,5,6,7,8,10,11,12,15,16,17,18,19,20,21,27,42]
        ama20pt=np.array([5,0,0,1,2,-1,3,4,6,7,5,10,5,5,12,13,14,15,16,17])
        return amass20,ama20pt
    def tranAndBias():
        '''
        将世界坐标转换为以摄像机为原点，x右，y下，z里的坐标，单位mm
        '''
        tranPG1=np.array([
            [0.0305190990118828,0.0631867211748541,-0.997534973252905],
            [0.999421825882832,0.0130345234848274,0.0314024703892722],
            [0.0149866121764903,-0.997916599453429,-0.0627523863353157]
            ])
        biasPG1=np.array([-177.231544221141,1030.55751094524,4999.31781497864])
        tranPG2=np.array([
            [0.998867199351677,-0.00492512595384636,-0.0473292847365708],
            [0.0456930239680428,-0.178391443589908,0.982898082414736],
            [-0.0132840362839212,-0.983947272971600,-0.177964317748699]
            ])
        biasPG2=np.array([-243.680254855334,672.573319492669,4050.87266768758])
        return tranPG1,biasPG1,tranPG2,biasPG2

class dataProcess:
    '''
    用于进行MoVi数据集预处理
    '''
    def videoToNumpy(videoPath,dtype='uint8',resize=(300,400),cut=0):

        cap=cv2.VideoCapture(videoPath)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if resize is None:
            videoData = np.empty((frameCount,3 ,frameHeight, frameWidth), np.dtype(dtype))
        else:
            videoData = np.empty((frameCount,3 ,resize[0], resize[1]), np.dtype(dtype))
        fc=0
        ret = True
        while (fc < frameCount  and ret):
            ret, c_a = cap.read()
            if cut==1:
                _shape=c_a.shape
                _b=int((_shape[1]-_shape[0])/2)
                c_a=c_a[:,_b:_b+_shape[0],:]
            if resize is not None:
                c_a=cv2.resize(c_a,(resize[1],resize[0]))
            c_a=c_a[...,[2,1,0]]
            videoData[fc]=c_a.swapaxes(0,2).swapaxes(1,2)
            fc += 1
        return videoData

    def getMoVidata(subject=86,PG=1,fps=30,dataPath='D:/1Datasets/MoVi',resize=(300,400),tran=1,cut=0):

        assert fps in [1,5,10,15,30]#检测fps是否可用
        assert PG in [1,2]#检测相机是否可用   正面和侧面
        #获取旋转矩阵和偏移矩阵
        t1,b1,t2,b2=MoViConst.tranAndBias()
        interval = int(30/fps)
        videoPath=dataPath+'/f_video/F_PG'+str(PG)+'_Subject_'+str(subject)+'_L.avi'
        v3dPath=dataPath+'/f_v3d/F_v3d_Subject_'+str(subject)+'.mat'
        amassPath=dataPath+'/f_mass_mat/F_amass_Subject_'+str(subject)+'.mat'
        video30=dataProcess.videoToNumpy(videoPath=videoPath,resize=resize,cut=cut)
        for action in range(21):#movi数据集有21个动作
            #每个动作的amass#将120帧的动作变为30帧#将数据类型由float64变为float32
            amass30=np.array(tools.mat2dict(amassPath)['move_'+str(action)]['jointsLocation_amass'])[0:-1:4,:,:].astype('float32')
            actionLen=amass30.shape[0]
            if tran==1 and PG==1:
                amass30=(np.dot(amass30,t1)+b1[np.newaxis,np.newaxis,...]).astype('float32')
            if tran==1 and PG==2:
                amass30=(np.dot(amass30,t2)+b2[np.newaxis,np.newaxis,...]).astype('float32')
            # print('amass30:',amass30.shape,amass30.dtype)
            # print('action:',action,'len:',actionLen)
            [actionBegin,actionEnd]=np.array(MoViutils.mat2dict(v3dPath)['move']['flags30'][action,:])#获取视频（30fps）的对应动作开始的帧节点
            if action == 0:
                validVideoData=video30[actionBegin:actionBegin+actionLen:interval,...]
                validJointsData=amass30[::interval,...]
            else:
                validVideoData=np.concatenate((validVideoData,video30[actionBegin:actionBegin+actionLen:interval,...]),axis=0)
                validJointsData=np.concatenate((validJointsData,amass30[::interval,...]),axis=0)
            # print('videolen',validVideoData.shape[0])
        return validVideoData, validJointsData, #validActionsData
    def outputMoViData(interval,subjectList=None,dataPath='D:/1Datasets/MoVi',resize=(300,400),tran=1,cut=0,PG=(1,2)):

        allVideo=None
        allJoints=None
        if subjectList is not None:
            outName='_'
            for j in subjectList:
                outName+=str(j)+'_'
            if resize is not None:
                outName+='Re_'
            if tran==1:
                outName+='Tr_'
            if cut==1:
                outName+='Ct_'
            outName+='PG'+str(PG)+'_'
            for i in subjectList:
                #正面相机
                if 1 in PG:
                    vi,jo=dataProcess.getMoVidata(subject=i,PG=1,fps=30,dataPath='D:/1Datasets/MoVi',resize=resize,tran=tran,cut=cut)
                    if allVideo is None:
                        allVideo=vi[::interval,...]
                    else:
                        allVideo=np.concatenate((allVideo,vi[::interval,...]),axis=0)
                    if allJoints is None:
                        allJoints=jo[::interval,...]
                    else:
                        allJoints=np.concatenate((allJoints,jo[::interval,...]),axis=0)
                #侧面相机
                if 2 in PG:
                    vi,jo=dataProcess.getMoVidata(subject=i,PG=2,fps=30,dataPath='D:/1Datasets/MoVi',resize=resize,tran=tran,cut=cut)
                    if allVideo is None:
                        allVideo=vi[::interval,...]
                    else:
                        allVideo=np.concatenate((allVideo,vi[::interval,...]),axis=0)
                    if allJoints is None:
                        allJoints=jo[::interval,...]
                    else:
                        allJoints=np.concatenate((allJoints,jo[::interval,...]),axis=0)
        outPath=dataPath+'/yyData/'+'MoViData_Subj'+outName[:-1]+'_Inte_'+str(interval)+'.npy'
        with open(outPath, 'wb') as f:
            np.save(f, allVideo)
            np.save(f, allJoints)
        return

    def loadYYDataToNumpy(fileName=None,dataPath='D:/1Datasets/MoVi/yyData'):

        if fileName is not None:
            with open(dataPath+'/'+fileName, 'rb') as f:
                allVideo = np.load(f)
                allJoints = np.load(f)
        return allVideo, allJoints
    def JointsNormalization(allJonits,mode=0):

        xmax=np.max(allJonits[...,0])
        xmin=np.min(allJonits[...,0])
        ymax=np.max(allJonits[...,1])
        ymin=np.min(allJonits[...,1])
        zmax=np.max(allJonits[...,2])
        zmin=np.min(allJonits[...,2])
        if mode==0:
            allJonits[...,0]=(allJonits[...,0]-(xmax/2+xmin/2))/(xmax-xmin)*2
            allJonits[...,1]=(allJonits[...,1]-(ymax/2+ymin/2))/(ymax-ymin)*2
            allJonits[...,2]=(allJonits[...,2]-(zmax/2+zmin/2))/(zmax-zmin)*2
            return allJonits
        if mode==1:
            allJonits[...,0]=(allJonits[...,0]-(xmax/2+xmin/2))/(xmax-xmin)*2
            allJonits[...,1]=(allJonits[...,1]-(ymax/2+ymin/2))/(ymax-ymin)*2
            allJonits[...,2]=(allJonits[...,2]-(zmax/2+zmin/2))/(zmax-zmin)*2
            scale=np.array([xmax-xmin,ymax-ymin,zmax-zmin])/2
            bias=np.array([-(xmax/2+xmin/2),-(ymax/2+ymin/2),-(zmax/2+zmin/2)])
            return allJonits,scale,bias
    '''
    对human3.6数据集进行操作
    '''
    def getSingleHM(sbj,act,subact,ca,interval=10,tarSize=(448,448),dataBasePath='D:/1Datasets/Human3.6M'):

        assert sbj in [1,5,6,7,8,9,11]
        assert act in range(2,17)
        assert subact in [1,2]
        assert ca in [1,2,3,4]
        #读取joints mat
        name='s_'+format(sbj,'02d')+'_act_'+format(act,'02d')+'_subact_'+format(subact,'02d')+'_ca_'+format(ca,'02d')
        print('开始获取:',name)    
        matPath=dataBasePath+'/images/'+name+'/h36m_meta.mat'
        matdata=sio.loadmat(matPath)
        allJoints=matdata['pose3d_world'][::interval,...]
        # print('allJoints大小:',allJoints.shape,'格式',allJoints.dtype,'images文件夹中的节点数据')
        #读取annotations相机信息
        jsonCamPath=dataBasePath+'/annotations/Human36M_subject'+str(sbj)+'_camera.json'
        camInfo=json.load(open(jsonCamPath))
        RMat=np.array(camInfo[str(ca)]['R'])
        TMat=np.array(camInfo[str(ca)]['t'])
        #以相机建立坐标系
        allJoints=np.dot(allJoints,RMat.T)+TMat
        #获得图片信息
        allVideos=np.empty(shape=(allJoints.shape[0],3,tarSize[0],tarSize[1]),dtype='uint8')
        # print('allVideos大小:',allVideos.shape,'格式',allVideos.dtype)
        for i in range(allJoints.shape[0]):
            imagePath=dataBasePath+'/images/'+name+'/'+name+'_'+format(i*interval+1,'06d')+'.jpg'
            # print(imagePath)
            image=cv2.imread(imagePath)
            image=cv2.resize(image,(tarSize[1],tarSize[0]))
            image=image[...,[2,1,0]]
            allVideos[i]=image.swapaxes(0,2).swapaxes(1,2)
        return allVideos,allJoints
    class HM36Const:
        '''
        节点信息
        0   胯下 7
        1   右臀 0
        2   右膝盖1
        3   右脚后跟2
        4   左臀0
        5   左膝盖4
        6   左脚后跟5
        7   后背中心-1
        8   脖子后下方7
        9   鼻子8
        10  额头9
        11  左肩膀8
        12  左肘11
        13  左手12
        14  右肩膀8
        15  右肘14
        16  右手15
        '''
        def getParent():
            parent=np.array(
                [7,0,1,2,0,4,5,-1,7,8,9,8,11,12,8,14,15]
            )
            return parent
    def outputHM36(subjects=[],interval=10,tarSize=(448,448),dataBasePath='D:/1Datasets/Human3.6M',subactList=[1,2],outputPath='D:/1Datasets/Human3.6M'):
        '''
        导出HM36数据集
        '''
        if outputPath is None:
            outputPath=dataBasePath+'/yyData/HM36_'+str(subjects)+'_Inte_'+str(interval)+'.npy'
        else:
            outputPath=outputPath+'/yyData/HM36_'+str(subjects)+'_Inte_'+str(interval)+'.npy'
        allVideos=None
        allJoints=None
        av=[]
        aj=[]
        for sbj in subjects:
            for act in range(2,17):
                for subact in subactList:
                    for ca in [1,2,3,4]:
                        try:
                            ca_videos,ca_joints=dataProcess.getSingleHM(sbj,act,subact,ca,interval,tarSize,dataBasePath)
                            av.append(ca_videos)
                            aj.append(ca_joints)
                            # if allVideos is None:
                            #     allVideos = ca_videos
                            #     allJoints = ca_joints
                            # else:
                            #     allVideos=np.concatenate((allVideos,ca_videos),axis=0)
                            #     allJoints=np.concatenate((allJoints,ca_joints),axis=0)
                        except:
                            print(sbj,act,subact,ca,'!!!!!!!!!!!!!')
        allVideos=np.concatenate(av,axis=0)
        allJoints=np.concatenate(aj,axis=0)
        with open(outputPath,'wb') as f:
            np.save(f,allVideos)
            np.save(f,allJoints)
        return
    def MPJPE(yHat,y):

        return ((((yHat-y)**2).sum(-1))**0.5).mean(-1).mean(-1)
class test:
    '''
    测试函数
    '''
    def test_movi_v3d():
        '''
        用于测试MoVi数据集的v3d数据
        实测发现v3d数据存在某些点记录错误的情况
        '''

        videoPath='D:/1Datasets/MoVi/f_video/F_PG1_Subject_86_L.avi'
        v3dPath='D:/1Datasets/MoVi/f_v3d/F_v3d_Subject_86.mat'


        a=np.array([[[0.1,0.1,0.1],
                    [0.5,0.5,0.5],
                    [0.5,0.5,0.1],
                    [0.9,0.9,0.9]]])*1000
        ap=np.array([1,2,3,-1])


        ml= tools.mat2dict(v3dPath)['move']['virtualMarkerLocation']
        ml=ml[0::20,:,:]
        mp=tools.mat2dict(v3dPath)['move']['virtualMarkerParent']
        mp=np.append(-1,mp)
        print(mp.shape)


        tools.showMotion(markerlocation=ml,parent=mp,fps=6,save=0)

        return

    def test_movi_amass(num):

        connections_amassPath='D:/1Datasets/MoVi/f_mass_mat/connections_amass.mat'
        parent=np.array([ 0,  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 10, 13,
                14, 15, 17, 18, 19, 20, 21, 23, 24, 21, 26, 27, 21, 29, 30, 21,
                32, 33, 21, 35, 36, 22, 38, 39, 22, 41, 42, 22, 44, 45, 22, 47,
                48, 22, 50, 51])-1
        amassPath='D:/1Datasets/MoVi/f_mass_mat/F_amass_Subject_86.mat'
        amass=np.array(tools.mat2dict(amassPath)['move_'+str(num)]['jointsLocation_amass'])[0:-1:2,:,:]

        tools.showMotion(markerlocation=amass,parent=parent,fps=60,save=0)
        return

    def test_movi_videoAndAmass(action=2,subject=86):

        #在此示例中，动作共有21个
        videoPath='D:/1Datasets/MoVi/f_video/F_PG1_Subject_'+str(subject)+'_L.avi'
        v3dPath='D:/1Datasets/MoVi/f_v3d/F_v3d_Subject_'+str(subject)+'.mat'
        #connections_amassPath='D:/1Datasets/MoVi/f_mass_mat/connections_amass.mat'
        videoCap=cv2.VideoCapture('D:/1Datasets/MoVi/f_video/F_PG1_Subject_'+str(subject)+'_L.avi')
        amassPath='D:/1Datasets/MoVi/f_mass_mat/F_amass_Subject_'+str(subject)+'.mat'

        parent=np.array([ 0,  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 10, 13,
                14, 15, 17, 18, 19, 20, 21, 23, 24, 21, 26, 27, 21, 29, 30, 21,
                32, 33, 21, 35, 36, 22, 38, 39, 22, 41, 42, 22, 44, 45, 22, 47,
                48, 22, 50, 51])-1
        amass=np.array(tools.mat2dict(amassPath)['move_'+str(action)]['jointsLocation_amass'])[0:-1:4,:,:]
        [actionBegin,actionEnd]=np.array(tools.mat2dict(v3dPath)['move']['flags30'][action,:])
        tools.showMotionAndVideo(markerlocation=amass,parent=parent,fps=30,videoCap=videoCap,videoBegin=actionBegin,save=1,show=0)
        return

class Timer:
    """
    取自d2l
    Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class myNN:
    ca_outputONNXPath='C:/Users/yuanying/Desktop/testonnx.onnx'
    class ResNetBlock(nn.Module):

        def __init__(self,inChannels,outChannels,stride=1,oneConv=False):
            super().__init__()
            self.conv1 = nn.Conv2d(inChannels, outChannels,
                                kernel_size=3, padding=1,stride=stride,bias=False)
            self.conv2 = nn.Conv2d(outChannels, outChannels,
                                kernel_size=3, padding=1,bias=False)
            if not inChannels==outChannels:
                self.conv3=nn.Conv2d(inChannels,outChannels,
                                    kernel_size=1,stride=stride,bias=False)
            else:
                if oneConv:
                    self.conv3=nn.Conv2d(inChannels,outChannels,
                        kernel_size=1,stride=stride,bias=False)
                else: 
                    self.conv3=None
            self.bn1 = nn.BatchNorm2d(outChannels)
            self.bn2 = nn.BatchNorm2d(outChannels)
            # self.bn3 = nn.BatchNorm2d(outChannels)
        def forward(self, X):

            Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
            if self.conv3 is not None:
                X = self.conv3(X)
            Y += X
            # return F.relu(self.bn3(Y))
            return F.relu(Y)
    class relu2d(nn.Module):
        def __init__(self,size):
            super().__init__()
            assert len(size)==2
            self.mat = nn.Parameter(torch.empty(*size),requires_grad=False)#.to('cuda')
            self.weight = nn.Parameter(torch.randn(1))#.to('cuda')
            self.bias = nn.Parameter(torch.randn(1))#.to('cuda')
            with torch.no_grad():
                for i in range(size[0]):
                    self.mat[i]=torch.arange(1,size[1]+1)+i
            
        def forward(self, X):
            # return (self.mat.data*X)*self.weight.data+self.bias.data
            return torch.add(torch.mul(torch.mul(self.mat, X),self.weight),self.bias)
    def getNet1_3_448_448():
        net1_3_448_448=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2),
            myNN.ResNetBlock(64,64),
            myNN.ResNetBlock(64,64),
            myNN.ResNetBlock(64,64),#这之后与另一支合并
            myNN.ResNetBlock(64,128,2),
            myNN.ResNetBlock(128,128),
            myNN.ResNetBlock(128,128),
            myNN.ResNetBlock(128,128),
            myNN.ResNetBlock(128,256,2),
            myNN.ResNetBlock(256,256),
            myNN.ResNetBlock(256,256),
            myNN.ResNetBlock(256,256),
            myNN.ResNetBlock(256,256),
            myNN.ResNetBlock(256,256),
            myNN.ResNetBlock(256,512,2),
            myNN.ResNetBlock(512,512),
            myNN.ResNetBlock(512,512),#结束resnet
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
            nn.Conv2d(in_channels=1024,out_channels=672,kernel_size=1),
            nn.BatchNorm2d(672),
            nn.ReLU()
        )
        return net1_3_448_448
    def init_weights(m):
        '''
        初始化
        net.apply(init_weights)
        '''
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    class tensorDataset(data.Dataset):

        def __init__(self,images,targets):
            super().__init__()
            self.images=images
            self.targets=targets
        def __getitem__(self, index):
            return self.images[index],self.targets[index]
        
        def __len__(self):
            return self.images.size(0)
    def getNetV1_1():
        net=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=7,padding=3,stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            myNN.ResNetBlock(6,6),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2),

            nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            myNN.ResNetBlock(12,12),
            nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            myNN.ResNetBlock(24,24),

            nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            myNN.ResNetBlock(48,48),

            nn.Conv2d(in_channels=48,out_channels=20,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            myNN.ResNetBlock(20,20),

            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=7,padding=3,stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            myNN.ResNetBlock(20,20),
            myNN.ResNetBlock(20,20),
            nn.Flatten(start_dim=2),
            nn.Linear(130,65),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(65,30),
            nn.BatchNorm1d(20),
            nn.Sigmoid(),
            nn.Linear(30,3),
            nn.BatchNorm1d(20),
            nn.Sigmoid(),
        )
        return net
    class CoordRegressionNetwork(nn.Module):
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
    def getResnetBackboneSpec(layersNum):
        '''
        获取经典resnet骨干网络参数
        18,34,50,101,152
        '''
        assert layersNum in [18,34,50,101,152]
        # Specification
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
                    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
                    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
                    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
                    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        return resnet_spec[layersNum]
    class ResNetBackbone(nn.Module):
        '''
        返回resnet骨干网络
        自带初始化
        '''
        def __init__(self, block, layers, in_channel=3):
            self.inplanes = 64
            super(myNN.ResNetBackbone, self).__init__()
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
                # layers.append(nn.BatchNorm2d(self.inplanes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x
    class ChannelMaxPooling(nn.Module):
        '''
        通道最大池化
        在第dim维度进行 默认为1
        '''
        def __init__(self, dim=1):
            super().__init__()
            self.dim=dim
        def forward(self,x):
            return (torch.max(x,dim=self.dim,keepdim=True))[0]
    def changeLr(opt,targetLr):
        for param_group in opt.param_groups:
            param_group["lr"] = targetLr
class Animator:
    """
    来自d2l
    For plotting data in animation.
    
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: Animator.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)
        # display.display(self.fig)
        # display.clear_output(wait=True)
    
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.
        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()
class drawSomeLines:
    '''
    一般使用draw（）
    mode==1 隔点画
    mode==2 间隔画，但是取平均值
    '''
    def __init__(self,num,xlim=None,ylim=None,interval=1,mode=1):
        self.data = [([],[]) for i in range(num)]
        self.last = None
        self.color = ['r','g','b']
        self.label = ['trainLoss','evalLoss','other']
        self.begin = True
        self.interval=interval
        self._intv=interval
        self.mode=mode
        self._data=None
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
    def add(self,*xy):
        assert len(xy)==len(self.data)
        for i in range(len(self.data)):
            if xy[i] is not None:
                self.data[i][0].append(xy[i][0])
                self.data[i][1].append(xy[i][1]) 
        
    def show(self):
        plt.cla()
        for i in range(len(self.data)):
            plt.plot(*self.data[i])
        plt.draw()
        plt.pause(0.00001)
    
    def _add(self,xy):
        if self._data is None:
            self._data=[([],[]) for i in range(len(xy))]
        
        for i in range(len(xy)):
            self._data[i][0].append(xy[i][0])
            self._data[i][1].append(xy[i][1])
    def _getAvg(self,sizeLike):
        '''
        检查xy形状
        获得self._data均值
        '''
        Avg=sizeLike
        if self._data is not None:
            for i in range(len(sizeLike)):
                Avg[i][0]=sum(self._data[i][0])/len(self._data[i][0])
                Avg[i][1]=sum(self._data[i][1])/len(self._data[i][1])
        return Avg
    def draw(self,*xy):
        if self.mode==1:
            if self._intv==self.interval:
                if self.last is None:
                    self.last = xy
                else:
                    for i in range(len(xy)):
                        plt.plot([self.last[i][0],xy[i][0]],[self.last[i][1],xy[i][1]],self.color[i],label=self.label[i])
                        if self.begin:
                            plt.legend()
                    self.begin=False
                    plt.draw()
                    plt.pause(0.1)
                    self.last = xy
                self._intv=1
            else:
                self._intv+=1
        elif self.mode==2:
            self._add(xy)
            if self._intv==self.interval:
                Avg=self._getAvg(sizeLike=xy)
                if self.last is None:
                    self.last = Avg
                else:
                    for i in range(len(xy)):
                        plt.plot([self.last[i][0],Avg[i][0]],[self.last[i][1],Avg[i][1]],self.color[i],label=self.label[i])
                        if self.begin:
                            plt.legend()
                    self.begin=False
                    plt.draw()
                    plt.pause(0.1)
                    self.last = Avg
                self._intv=1
                self._data=None
            else:
                self._intv+=1
