import cv2
import numpy as np
import os
from random import shuffle
from PIL import Image
# from BK.MeshNetLayer import *
#from Lib3D.MeshModel import *
#from Lib3D.Rasterization import  *
#from Lib3D.Util2D import *
#from Lib3D.Util3D import *
from keras import optimizers
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.callbacks import ModelCheckpoint
import utility as ut


def final_pred(y_true, y_pred):
    # y_cont=np.concatenate(y_pred,axis=1)
    return y_pred

def drawPTS(img, pts,imgW=128):
    imgShow = img.copy()
    draw = ImageDraw.Draw(imgShow)
    PD2DList, PD2D = getPTSlist(pts, imgSize=imgW)
    draw.point(PD2DList, fill='red')
    draw.rectangle([(PD2D[68,0],PD2D[68,1]),(PD2D[69,0],PD2D[69,1])], fill=None, outline='red')
    return imgShow

def transformBB(img,pts, type='nus68',imgW=128,RotAngle=0):
    crop_size = img.shape[0]
    assert(img.shape[0]==img.shape[1])

    tform_shift = tf.SimilarityTransform(translation=[crop_size/2, crop_size/2])
    tform_rand = tf.SimilarityTransform(rotation=np.deg2rad(RotAngle))
    tform_shift_inv = tf.SimilarityTransform(translation=[-crop_size/2, -crop_size/2])
    tform = tform_shift_inv+tform_rand+tform_shift

    warp_img = tf.warp(np.asarray(img/255.0), tform)
    pts = (tform.inverse(pts*imgW+imgW/2)-imgW/2)/imgW
    warp_img = warp_img*255
    warp_img = warp_img.astype('uint8')
    warp_img = Image.fromarray(warp_img)

    return (warp_img,pts)

def DataGenBB(DataStrs, BatchSize,train_start,train_end,imSize=128):

    InputData = np.zeros([BatchSize,imSize,imSize,3],dtype=np.float32)
    InputLabel = np.zeros([BatchSize,4],dtype=np.float32)
    InputRot = np.zeros([BatchSize,3],dtype=np.float32)
    # shuffle(DataStrs)
    InputNames = []
    # MaxIters = len(DataStrs) / BatchSize
    # for Mi in range(MaxIters):
        # train_start = BatchSize*Mi
        # train_end = train_start+BatchSize
    counter = 0
    for i in range(train_start,train_end):
        strLine = DataStrs[i]
        strCells = strLine.rstrip(' \n').split(' ')
        imgName = strCells[0]
        labels = np.array(strCells[1:]).astype(np.float)
        labelsPTS=labels[:136].reshape([68,2])
        print imgName
        img = cv2.imread(imgName)
        if img != None:
            x, y = ut.unpackLandmarks(labelsPTS)

            rotateImg, rotateX, rotateY = ut.rotate(img, x, y)
            print "rotateX[:10]: ", rotateX[:10]
            # resizeImg, resizeX, resizeY = ut.resize(img, x, y, random = True)
            # plotOriginal = ut.plotLandmarks(img, x, y, ifRescale = True, ifReturn = True)
            plotRotate = ut.plotLandmarks(rotateImg, rotateX, rotateY, ifReturn = True)
            # plotRotate = ut.plotLandmarks(rotateImg, rotateX, rotateY, ifRescale = True, ifReturn = True)
            # plotResize = ut.plotLandmarks(resizeImg, resizeX, resizeY, ifRescale = True, ifReturn = True)

            cv2.imwrite('testOriginal' + str(counter) + '.jpg', img)
            cv2.imwrite('testRotate' + str(counter) + '.jpg', rotateImg)
            # cv2.imwrite('testResize' + str(counter) + '.jpg', resizeImg)
            
            # cv2.imwrite('plotOriginal' + str(counter) + '.jpg', plotOriginal)
            cv2.imwrite('plotRotate' + str(counter) + '.jpg', plotRotate)
            # cv2.imwrite('plotResize' + str(counter) + '.jpg', plotResize)

            counter += 1
        else:
            print "cannot find: ", imgName

    #     im = cv2.resize(cv2.imread(imgName), (imSize, imSize)).astype(np.float32)
    #     im = im[..., np.array([2, 1, 0])]
    #     # Rot, Scale, T, theta= GetRTS(labelsPTS, MeanShape)
    #     # ims = Image.fromarray(im.astype(np.uint8))
    #     # ims.save('./org.jpg')
    #     RotAngle = (random.randint(0,2)-1)*90
    #     imgRot, PTSRot=transformBB(im,labelsPTS,RotAngle=RotAngle)

    #     Rot, Scale, T, theta= GetRTS(PTSRot, MeanShape)
    #     # print np.rad2deg(theta)
    #     Dlabel=np.zeros([1,3],dtype=int);
    #     Dlabel[0,1]=1;
    #     if theta<np.deg2rad(-55):
    #         Dlabel[0,2]=1
    #         Dlabel[0,1]=0
    #     if theta>np.deg2rad(55):
    #         Dlabel[0,0]=1
    #         Dlabel[0,1]=0
    #     mins = np.min(PTSRot,axis=0)
    #     maxs = np.max(PTSRot,axis=0)

    #     InputData[count,...]=imgRot.copy()
    #     InputLabel[count,...]=np.array([mins[0],mins[1],maxs[0],maxs[1]])

    #     InputRot[count,...]=Dlabel
    #     InputNames.append(imgName)
    #     count+=1
    #     # PtsB = np.concatenate([PTSRot,InputLabel[count,...].reshape(2,2)],axis=0)
    #     # imgDraw=drawPTS(imgRot,PtsB,imgW=128)
    #     # imgDraw.save('./tmp.jpg')
    # return InputData,InputLabel,InputRot,InputNames








# TN = TextNet('./MatBS/shape_0.obj', imgW=256)
TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
# TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'

FTr = open(TrainPath,'r')
DataTr = FTr.readlines()
# FTe = open(TestPath,'r')
# DataTe = FTe.readlines()


# DataLabels = np.zeros([len(DataTr) / 10, 136])
# for i in range(len(DataTr) / 10):
#     # i=0
#     crtLabel = DataTr[i]
#     X = crtLabel.rstrip(' \n').split(' ')[1:137]
#     DataLabels[i, :] = np.asarray(X).astype(np.float32)
# Mean = np.mean(DataLabels,axis=0)
# np.savetxt('./MeanShape.txt',Mean)
from keras import callbacks

import shutil



#BBNet = BBFullNet(weights_path='./BBNet/BBNet_V1.h5',imgW=128)
#sgdBB = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
#BBNet.compile(loss={'BB_RCT':'mean_squared_error','Img_Rot':'categorical_crossentropy'}, loss_weight=[1,10],metrics=['accuracy', final_pred],optimizer=sgdBB)
#BBNet.summary()

# Tmp=np.loadtxt('./MeanShape.txt')
# MeanShape = Tmp[:136].reshape([68,2])
# MeanShape = None




TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
# TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'
FTr = open(TrainPath,'r')
DataTr = FTr.readlines()
# FTe = open(TestPath,'r')
# DataTe = FTe.readlines()
# TrData,TrLabel=load_train_data(DataTr,0,5,5)

batch_size=16
TrNum = len(DataTr)
# TeNum = len(DataTe)
MaxIters = TrNum/batch_size
# MaxTestIters = TeNum/batch_size

SamplePerEpoch=len(DataTr)/100
NumEpoch = 200;


def train_on_batch(nb_epoch):
    for e in range(nb_epoch):
        # if e>0:
        shuffle(DataTr)
        iterTest=0
        for iter in range (MaxIters):
            train_start=iter*batch_size
            train_end = (iter+1)*batch_size
            DataGenBB(DataTr,batch_size,train_start=train_start, train_end=train_end,imSize=128)
            # X_batch, label_BB, label_rot, Z_Names = DataGenBB(DataTr,batch_size,train_start=train_start, train_end=train_end,imSize=128)
            print "finish iteration: ", iter
            # lossBB,tras,lossRot,tras,PredBB,tras,PredRot= BBNet.train_on_batch(X_batch,[label_BB,label_rot])
            # if iter%100==0:
            #     print 'iter ', iter,'Traing loss: ', lossBB, lossRot
            #     test_start = iterTest * batch_size
            #     test_end = (iterTest + 1) * batch_size
            #     X_batch_T, label_BB_T, label_rot_T, Z_Names_T= DataGenBB(DataTe, batch_size, MeanShape=MeanShape,
            #                                                       train_start=test_start, train_end=test_end,
            #                                                       imSize=128)
            #     lossBBT, tras, lossRotT,tras, PredBBT, tras, PredRotT = BBNet.evaluate(X_batch_T,[label_BB_T,label_rot_T])
            #     print 'iter ', iter,'Testing loss: ', lossBBT, lossRotT
            #     iterTest+=batch_size
            #     iterTest%=MaxTestIters

            #     img = X_batch_T[0,...]
            #     img2Draw=Image.fromarray(img.astype(np.uint8))
            #     imDraw=ImageDraw.Draw(img2Draw)
            #     PD2D = PredBBT[0,:]*128+64
            #     imDraw.rectangle([(PD2D[0], PD2D[1]), (PD2D[2], PD2D[3])], fill=None, outline='red')
            #     RotScore = PredRotT[0,:]
            #     RotType=np.where(RotScore==np.max(RotScore))[0]
            #     img2Draw.save('./BBNet/Test_tmp_%d_%d.jpg' % (RotType, np.where(label_rot_T[0, :] == 1)[0]))

            #     img = X_batch[0,...]
            #     img2Draw=Image.fromarray(img.astype(np.uint8))
            #     imDraw=ImageDraw.Draw(img2Draw)
            #     PD2D = PredBB[0,:]*128+64
            #     imDraw.rectangle([(PD2D[0], PD2D[1]), (PD2D[2], PD2D[3])], fill=None, outline='red')
            #     RotScore = PredRot[0,:]
            #     RotType=np.where(RotScore==np.max(RotScore))[0]
            #     img2Draw.save('./BBNet/Train_tmp_%d_%d.jpg'%(RotType,np.where(label_rot[0,:]==1)[0]))
            #     print label_rot[0,...], label_rot_T[0,...]
            # if iter%2000==0:
            #     BBNet.save('./BBNet/BBNet_V1.h5')

# sgdBB = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
# train_on_batch(2)

# sgdBB = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# train_on_batch(1)

# sgdBB = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
train_on_batch(1)
