import torch 
import numpy as np
import onnx 
import onnxruntime
import cv2
import yy
'''
用于临时测试输出的网络

'''
ONNXPath='./output/论文testnet.onnx'
# ONNXPath='C:/Users/yuanying/Desktop/outputONNX（movi res101 ）.onnx'
# ONNXPath='C:/Users/yuanying/Desktop/outputONNX（movi res101 ）.onnx'
mode=1 #1 从测试集获取数据 2 从自己照片获取数据

if mode ==1:
    testVideo, testJoint=yy.dataProcess.loadYYDataToNumpy(fileName='MoViData_Subj_87_Re_Tr_Ct_PG[1]_Inte_3.npy',dataPath='D:/1Datasets/MoVi/yyData')
    # testVideo, testJoint=yy.dataProcess.loadYYDataToNumpy(fileName='HM36_[11]_Inte_50.npy',dataPath='D:/1Datasets/Human3.6M/yyData')
    sess = onnxruntime.InferenceSession(ONNXPath)
    inputs = sess.get_inputs()
    while True:
        index=np.random.randint(0,testVideo.shape[0])
        print(index)
        img = testVideo[index,...].astype(np.float32)/255.0
        #模型
        out= sess.run(None,{
            inputs[0].name:img[np.newaxis,...],
        })
        yy.tools.showMotionAndVideo(markerlocation=yy.tools.changeCoord(out[0]),parent=yy.MoViConst.getAmass20()[1],fps=1,video=img[np.newaxis,...],azim=-60,elev=-160,normlization=0)
        # yy.tools.showMotionAndVideo(markerlocation=yy.tools.changeCoord(out[0]),parent=yy.dataProcess.HM36Const.getParent(),fps=1,video=img[np.newaxis,...],azim=-90,elev=-179,normlization=0)

elif mode ==2 :
    for i in range(15):
        picturePath='./testImages/test/test'+str(i)+'.png'
        img=cv2.imread(picturePath)
        img=cv2.resize(img,(448,448))
        img=img[...,[2,1,0]]
        img=img.swapaxes(0,2).swapaxes(1,2)
        #4:3转1:1
        # img=np.concatenate((np.random.randint(size=(3,50,400),high=256,low=0,dtype='uint8'),img,np.random.randint(size=(3,50,400),high=256,low=0,dtype='uint8')),axis=1)

        img = img.astype(np.float32)/255.0
        print('输入',img.shape,type(img))
        img = img[np.newaxis,...]
        #模型
        sess = onnxruntime.InferenceSession(ONNXPath)
        inputs = sess.get_inputs()
        out= sess.run(None,{
            inputs[0].name:img,
        })
        print('输出',out[0].shape)
        # exit()
        yy.tools.showMotionAndVideo(markerlocation=yy.tools.changeCoord(out[0]),parent=yy.MoViConst.getAmass20()[1],fps=1,video=img,azim=-90,elev=-179,normlization=0)
