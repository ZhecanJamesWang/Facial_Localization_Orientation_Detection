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
import random
from keras import callbacks
import shutil
# import model as m
import vgg16Modified as m


debug = True

def final_pred(y_true, y_pred):
    # y_cont=np.concatenate(y_pred,axis=1)
    return y_pred

def DataGenBB(DataStrs, BatchSize,train_start,train_end,imSize = 128):

    generateFunc = ["rotate", "resize"]

    InputData = np.zeros([BatchSize,imSize,imSize,3],dtype=np.float32)
    InputLabel = np.zeros([BatchSize,7],dtype=np.float32)

    # print "InputData.shape: ", InputData.shape
    # print "InputLabel.shape: ", InputLabel.shape

    InputNames = []
    count = 0
    for i in range(train_start,train_end):
        strLine = DataStrs[i]
        strCells = strLine.rstrip(' \n').split(' ')
        imgName = strCells[0]
        labels = np.array(strCells[1:]).astype(np.float)
        labelsPTS=labels[:136].reshape([68,2])

        # if debug:
        #     print "imgName: ", imgName
        img = cv2.imread(imgName)

        if img != None:
            img = cv2.resize(img,(128, 128))
            # print "img.shape: ", img.shape
            (w, h, _) = img.shape
            x, y = ut.unpackLandmarks(labelsPTS)
            
            tag = random.choice(generateFunc)

            # if tag == "rotate":
            newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
            # elif tag == "resize":
            #     newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
            # else:
            #     raise "not existing function"

            if debug:
                plotOriginal = ut.plotLandmarks(img, x, y, ifReturn = True)
                plotNew = ut.plotLandmarks(newImg, newX, newY, ifReturn = True)

                cv2.imwrite('testOriginal' + str(count) + '.jpg', img)
                cv2.imwrite('testNew' + str(count) + '.jpg', newImg)        
                cv2.imwrite('plotOriginal' + str(count) + '.jpg', plotOriginal)
                cv2.imwrite('plotNew' + str(count) + '.jpg', plotNew)

            
            newPTS = np.asarray(ut.packLandmarks(newX, newY))
            # print "newPTS: ", newPTS.shape


            xMin = min(newX) if min(newX) >= 0 else 0
            yMin = min(newY) if min(newY) >= 0 else 0
            xMax = max(newX) if max(newX) <= w else w
            yMax = max(newY) if max(newY) <= h else h
            xMean = np.mean(newX) if np.mean(newX) > 0 and np.mean(newX) < w else None
            yMean = np.mean(newY) if np.mean(newY) > 0 and np.mean(newY) < h else None
            edge = max(yMax - yMin, xMax - xMin)
            
            # print "len(InputData): ", len(InputData)
            InputData[count,...] = newImg
            labels = np.array([newPTS[27][0], newPTS[27][1], newPTS[8][0], 
                newPTS[8][1], xMean, yMean, edge])
            InputLabel[count,...] = labels
            InputNames.append(imgName)

            labelImg = ut.plotTarget(newImg, labels)
            cv2.imwrite('labelImg' + str(count) + '.jpg', labelImg)

            # print "count: ", count
            count += 1

        else:
            print "cannot find: ", imgName




        # PtsB = np.concatenate([PTSRot,InputLabel[count,...].reshape(2,2)],axis=0)
        # imgDraw=drawPTS(imgRot,PtsB,imgW=128)
        # imgDraw.save('./tmp.jpg')
    return InputData, InputLabel, np.asarray(InputNames)








# TN = TextNet('./MatBS/shape_0.obj', imgW=256)
TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
# TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'

FTr = open(TrainPath,'r')
DataTr = FTr.readlines()

# print "DataTr: ", type(DataTr)
# print len(DataTr)
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


#BBNet = BBFullNet(weights_path='./BBNet/BBNet_V1.h5',imgW=128)
#sgdBB = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
#BBNet.compile(loss={'BB_RCT':'mean_squared_error','Img_Rot':'categorical_crossentropy'}, loss_weight=[1,10],metrics=['accuracy', final_pred],optimizer=sgdBB)
#BBNet.summary()

# Tmp=np.loadtxt('./MeanShape.txt')
# MeanShape = Tmp[:136].reshape([68,2])
# MeanShape = None
# TrData,TrLabel=load_train_data(DataTr,0,5,5)

batch_size=16
TrNum = len(DataTr)
# TeNum = TrNum
# TeNum = len(DataTe)
MaxIters = TrNum/batch_size
# MaxTestIters = TeNum/batch_size

model = m.model(input_shape=(128, 128, 3))

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', final_pred])
model.summary()

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))


def train_on_batch(nb_epoch):
    testCount = 0
    trainCount = 0

    for e in range(nb_epoch):
        # if e>0:
        shuffle(DataTr)
        iterTest=0
        for iter in range (MaxIters):
            train_start=iter*batch_size
            train_end = (iter+1)*batch_size
            print "train_start: ", train_start
            print "train_end: ", train_end
            X_batch, label_BB, Z_Names = DataGenBB(DataTr,batch_size,train_start=train_start, train_end=train_end, imSize = 128)

            # if debug:
            #     print "X_batch.shape: ", X_batch.shape
            #     print "label_BB.shape: ", label_BB.shape
            #     print "Z_Names.shape: ", Z_Names.shape
            #     print "finish iteration: ", iter

            loss, tras, pred = model.train_on_batch(X_batch,label_BB)

            labelImg = ut.plotTarget(X_batch[0], pred[0])
            cv2.imwrite('trainLabelImg' + str(trainCount) + '.jpg', labelImg)
            trainCount += 1

            print "****************************************************************************"
            print "loss, return on train: ", type(loss), loss
            print "loss.shape: ", loss.shape
            print "pred, return on train: ", type(pred)
            print "pred.shape: ", pred.shape
            print "tras, return on train: ", type(tras), tras
            print "tras.shape: ", tras.shape 


            if iter%10==0:
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                print 'iteration: ', iter
                test_start = iterTest * batch_size
                test_end = (iterTest + 1) * batch_size
                X_batch_T, label_BB_T, Z_Names_T= DataGenBB(DataTr, batch_size, train_start=test_start, train_end=test_end, imSize = 128)
                loss, tras, pred = model.evaluate(X_batch_T,label_BB_T)

                labelImg = ut.plotTarget(X_batch_T[0], pred[0])
                cv2.imwrite('testLabelImg' + str(testCount) + '.jpg', labelImg)
                testCount += 1

                print "========================================================================="
                print "loss, return on test: ", type(loss), loss
                print "loss.shape: ", loss.shape
                print "pred, return on test: ", type(pred)
                print "pred.shape: ", pred.shape
                print "tras, return on test: ", type(tras), tras
                print "tras.shape: ", tras.shape 

                print 'iter ', iter,'Testing loss: ', loss
                iterTest+=batch_size
                # iterTest%=MaxTestIters

                # img = X_batch_T[0,...]
                # img2Draw=Image.fromarray(img.astype(np.uint8))
                # imDraw=ImageDraw.Draw(img2Draw)
                # PD2D = PredBBT[0,:]*128+64
                # imDraw.rectangle([(PD2D[0], PD2D[1]), (PD2D[2], PD2D[3])], fill=None, outline='red')
                # RotScore = PredRotT[0,:]
                # RotType=np.where(RotScore==np.max(RotScore))[0]
                # img2Draw.save('./BBNet/Test_tmp_%d_%d.jpg' % (RotType, np.where(label_rot_T[0, :] == 1)[0]))

                # img = X_batch[0,...]
                # img2Draw=Image.fromarray(img.astype(np.uint8))
                # imDraw=ImageDraw.Draw(img2Draw)
                # PD2D = PredBB[0,:]*128+64
                # imDraw.rectangle([(PD2D[0], PD2D[1]), (PD2D[2], PD2D[3])], fill=None, outline='red')
                # RotScore = PredRot[0,:]
                # RotType=np.where(RotScore==np.max(RotScore))[0]
                # img2Draw.save('./BBNet/Train_tmp_%d_%d.jpg'%(RotType,np.where(label_rot[0,:]==1)[0]))
                # print label_rot[0,...], label_rot_T[0,...]

            # if iter%2000==0:
            #     BBNet.save('./BBNet/BBNet_V1.h5')

# sgdBB = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
# train_on_batch(2)

# sgdBB = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# train_on_batch(1)

# sgdBB = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
train_on_batch(1)
