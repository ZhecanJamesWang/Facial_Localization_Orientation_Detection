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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.callbacks import ModelCheckpoint
import utility as ut
import random
from keras import callbacks
import shutil
# import model as m
import vgg16Modified as m
import os


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
        # print "imgName: ", imgName
        img = cv2.imread(imgName)

        if img != None:
            img = cv2.resize(img,(128, 128))
            # print "img.shape: ", img.shape
            (w, h, _) = img.shape
            x, y = ut.unpackLandmarks(labelsPTS)

            # newImg, newX, newY = img, x, y            
            # tag = random.choice(generateFunc)

            # if tag == "rotate":
            newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
            # elif tag == "resize":
            # newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
            # else:
            #     raise "not existing function"

            if debug:
                plotOriginal = ut.plotLandmarks(img, x, y, ifReturn = True)
                plotNew = ut.plotLandmarks(newImg, newX, newY, ifReturn = True)

                cv2.imwrite(outputDir + 'testOriginal' + str(count) + '.jpg', img)
                cv2.imwrite(outputDir + 'testNew' + str(count) + '.jpg', newImg)        
                cv2.imwrite(outputDir + 'plotOriginal' + str(count) + '.jpg', plotOriginal)
                cv2.imwrite(outputDir + 'plotNew' + str(count) + '.jpg', plotNew)

            # print "before normalize: ", newX
            
            # normX = ut.normalize(newX)
            # normY = ut.normalize(newY)
            
            # print "after normalize: ", newX
            # print "after denormalize again: ", ut.deNormalize(newX)


            # normXMin = min(normX)
            # normYMin = min(normY)
            # normXMax = max(normX)
            # normYMax = max(normY)
            # normXMean = (normXMax + normXMin)/2.0
            # normYMean = (normYMax + normYMin)/2.0
            # normEdge = max(normYMax - normYMin, normXMax - normXMin)

            newXMin = min(newX)
            newYMin = min(newY)
            newXMax = max(newX)
            newYMax = max(newY)
            newXMean = (newXMax + newXMin)/2.0
            newYMean = (newYMax + newYMin)/2.0
            newEdge = max(newYMax - newYMin, newXMax - newXMin)
                        
            # print "newXMin: ", newXMin
            # print "newYMin: ", newYMin
            # print "newXMax: ", newXMax
            # print "newYMax: ", newYMax
            # print "newXMean: ", newXMean
            # print "newYMean: ", newYMean
            # print "newEdge: ", newEdge


            normX = ut.normalize(newX)
            normY = ut.normalize(newY)
            normPTS = np.asarray(ut.packLandmarks(normX, normY))
            normXMean, normYMean, normEdge = ut.normalize(newXMean), ut.normalize(newYMean), ut.normalize(newEdge)
            # print "newPTS: ", newPTS.shape

            # print "ut.deNormalize(normXMin): ", ut.deNormalize(normXMin)
            # print "ut.deNormalize(normYMin): ", ut.deNormalize(normYMin)
            # print "ut.deNormalize(normXMax): ", ut.deNormalize(normXMax)
            # print "ut.deNormalize(normYMax): ", ut.deNormalize(normYMax)
            # print "ut.deNormalize(normXMean): ",ut.deNormalize(normXMean)
            # print "ut.deNormalize(normYMean): ",ut.deNormalize(normYMean)
            # print "ut.deNormalize(normEdge): ", ut.deNormalize(normEdge)


            # print "len(InputData): ", len(InputData)
            InputData[count,...] = newImg
            labels = np.array([normPTS[27][0], normPTS[27][1], normPTS[8][0], 
                normPTS[8][1], normXMean, normYMean, normEdge])
            # print "input labels: ", labels
            InputLabel[count,...] = labels
            InputNames.append(imgName)

            # print "count: ", count
            count += 1

        else:
            print "cannot find: ", imgName




        # PtsB = np.concatenate([PTSRot,InputLabel[count,...].reshape(2,2)],axis=0)
        # imgDraw=drawPTS(imgRot,PtsB,imgW=128)
        # imgDraw.save('./tmp.jpg')

    return InputData, InputLabel, np.asarray(InputNames)






def train_on_batch(nb_epoch, MaxIters):
    if os.path.exists(modelDir)==False:
        os.mkdir(modelDir)
    testCount = 0
    trainCount = 0
    for e in range(nb_epoch):
        # if e>0:
        shuffle(DataTr)
        iterTest=0
        for iter in range (MaxIters):
            train_start=iter*batch_size
            train_end = (iter+1)*batch_size
            # print "train_start: ", train_start
            # print "train_end: ", train_end
            X_batch, label_BB, Z_Names = DataGenBB(DataTr,batch_size,train_start=train_start, train_end=train_end, imSize = 128)

            # print "X_batch.shape: ", X_batch.shape
            for i in range(batch_size):
                labels = label_BB[i]
                img = X_batch[i]
                # print "input ut.deNormalize(labels): ", ut.deNormalize(labels)
                # labelImg = ut.plotTarget(img, labels)
                labelImg = ut.plotTarget(img, ut.deNormalize(labels))
                cv2.imwrite(outputDir + 'inputTrainlabelImg' + str(trainCount) + '.jpg', labelImg)

            loss, tras, pred = model.train_on_batch(X_batch,label_BB)
            trainCount += 1

            if trainCount >= 20:
                trainCount = 0

            # print "****************************************************************************"
            print "loss, train: ", loss
            # print "loss.shape: ", loss.shape
            # print "pred, return on train: ", type(pred)
            # print "pred.shape: ", pred.shape
            # # print "pred #######: ", pred
            # print "tras, return on train: ", type(tras), tras
            # print "tras.shape: ", tras.shape 

            if iter%30 == 0:
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                print 'iteration: ', iter
                # labelImg = ut.plotTarget(X_batch[0], pred[0])
                labelImg = ut.plotTarget(X_batch[0], ut.deNormalize(pred[0]))
                cv2.imwrite(outputDir + 'predTrainLabelImg' + str(trainCount) + '.jpg', labelImg)
                


                test_start = iterTest * batch_size
                test_end = (iterTest + 1) * batch_size
                X_batch_T, label_BB_T, Z_Names_T= DataGenBB(DataTr, batch_size, train_start=test_start, train_end=test_end, imSize = 128)
                loss, tras, pred = model.evaluate(X_batch_T,label_BB_T)
                testCount += 1

                if testCount >= 20:
                    testCount = 0

                # labelImg = ut.plotTarget(X_batch_T[0], pred[0])
                labelImg = ut.plotTarget(X_batch_T[0], ut.deNormalize(pred[0]))
                cv2.imwrite(outputDir + 'predTestLabelImg' + str(testCount) + '.jpg', labelImg)

                print "========================================================================="
                print "loss, TEST: ", loss
                # print "loss.shape: ", loss.shape
                # print "pred, return on test: ", type(pred)
                # print "pred.shape: ", pred.shape
                # print "pred #######: ", pred
                # print "tras, return on test: ", type(tras), tras
                # print "tras.shape: ", tras.shape 

                print 'iter ', iter,'Testing loss: ', loss
                iterTest+=batch_size
                # iterTest%=MaxTestIters

            if iter%3000==0:
                model.save(modelDir + '/model%d.h5'%iter)





debug = True
outputDir = "./output03162017_01_0.001_only/"
modelDir = "./model03162017_01_0.001_only/"


# TN = TextNet('./MatBS/shape_0.obj', imgW=256)
TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
# TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'

FTr = open(TrainPath,'r')
DataTr = FTr.readlines()

# print "DataTr: ", type(DataTr)
# print len(DataTr)
# FTe = open(TestPath,'r')
# DataTe = FTe.readlines()


batch_size = 32
TrNum = len(DataTr)
# TeNum = TrNum
# TeNum = len(DataTe)
MaxIters = TrNum/batch_size
# MaxTestIters = TeNum/batch_size

model = m.model(input_shape=(128, 128, 3))

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', final_pred])
model.summary()
train_on_batch(1, MaxIters)

# sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', final_pred])
# model.summary()
# train_on_batch(1, MaxIters = 10000)

# sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', final_pred])
# model.summary()
# train_on_batch(1, MaxIters = 15000)



