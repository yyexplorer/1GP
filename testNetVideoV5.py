
import torch 
import numpy as np
import onnx 
import onnxruntime
import cv2
import yy
import torch.nn as nn 
from torch.nn import functional as F
import yydsntnn as dsntnn
'''
用于展示网络效果，并连续输出，并可视化
可计算MPJPE
'''
ONNXPath='./output/netV5_2_train_MOVI87.onnx'
ONNXPath='C:/Users/yuanying/Desktop/outputONNX.onnx'
# ONNXPath='./output/netV6_4.onnx'
normPath='C:/Users/yuanying/Desktop/HM156789NormScale_NormBias.npy'

dataset=2 #1为MOVI 2为HM36

if dataset==1:
    testVideo, testJoint=yy.dataProcess.loadYYDataToNumpy(fileName='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_3.npy',dataPath='D:/1Datasets/MoVi/yyData')
    amass52to20,amass20pt=yy.MoViConst.getAmass20()
    testJoint=testJoint[:,amass52to20,:]#转为amass20
if dataset==2:
    testVideo, testJoint=yy.dataProcess.loadYYDataToNumpy(fileName='HM36_[11]_Inte_50.npy',dataPath='D:/1Datasets/Human3.6M/yyData')

# 获取归一化参数
if 'netV5_2_train_MOVI87.onnx' in ONNXPath: 
    moviScale=np.array([1455.3582,1148.7917,928.0657])
    moviBias=np.array([143.54296875,14.82806396,-4623.52075195])
else:
    with open(normPath,'rb') as f :
        moviScale=np.load(f)
        moviBias=np.load(f)

print('testVideo',testVideo.shape)
print('testJoint',testJoint.shape)
interval=1
end=testJoint.shape[0]

joints=testJoint[0:end:interval,...]
sess = onnxruntime.InferenceSession(ONNXPath)
inputs = sess.get_inputs()
markerlocation=None

for img in testVideo[0:end:interval,...]:
    img = img.astype(np.float32)/255.0
    out= sess.run(None,{inputs[0].name:img[np.newaxis,...],})
    _ml=out[0]
    if markerlocation is None:
        markerlocation=_ml
    else:
        markerlocation=np.concatenate((markerlocation,_ml),axis=0)
print(yy.dataProcess.MPJPE(markerlocation*moviScale-moviBias,joints))
markerlocation=yy.tools.changeCoord(markerlocation*moviScale-moviBias)
exit()
if dataset==1:
    yy.tools.showMotionAndVideo(
        markerlocation=markerlocation,
        parent=yy.MoViConst.getAmass20()[1],
        fps=10,
        video=testVideo[0:end:interval,...],
        save=1,
        show=0,
        azim=-60,
        elev=-160,)
if dataset==2:
    yy.tools.showMotionAndVideo(
        markerlocation=markerlocation,
        parent=yy.dataProcess.HM36Const.getParent(),
        fps=5,
        video=testVideo[0:end:interval,...],
        save=1,
        show=0,
        azim=-60,
        elev=-160,)