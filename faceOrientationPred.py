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
import os


class faceOrientPred(object):
    """face orientation detection"""
    def __init__(self):

        self.init = True
        self.debug = True
        self.outputDir = "./03182017_01_output/"
        self.modelDir = "./03182017_01_model/"
        self.imSize = 128

        # TN = TextNet('./MatBS/shape_0.obj', imgW=256)
        TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
        # TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'

        FTr = open(TrainPath,'r')
        self.DataTr = FTr.readlines()
        print "type(DataTr): ", type(self.DataTr)
        print "len(DataTr): ", len(self.DataTr)

        shuffle(self.DataTr)
        self.DataTe = self.DataTr[:int(len(self.DataTr)*0.1)]
        self.DataTr = self.DataTr[int(len(self.DataTr)*0.1):]
        print "len(self.DataTr): ", len(self.DataTr)
        print "len(self.DataTe): ", len(self.DataTe)

        TrNum = len(self.DataTr)

        # FTe = open(TestPath,'r')
        # DataTe = FTe.readlines()
        TeNum = len(self.DataTe)


        self.batch_size = 32
        self.MaxIters = TrNum/self.batch_size
        self.MaxTestIters = TeNum/self.batch_size

        print "train data length:", TrNum
        print "test data length:", TeNum






    def final_pred(self, y_true, y_pred):
        # y_cont=np.concatenate(y_pred,axis=1)
        return y_pred




    def DataGenBB(self, DataStrs, train_start,train_end):
        generateFunc = ["original", "resize", "rotate", "brightnessAndContrast" ]
        # generateFunc = ["orginal", "resize"， "rotate", "mirror", "translate", "brightnessAndContrast" ]

        InputData = np.zeros([self.batch_size, self.imSize, self.imSize, 3], dtype = np.float32)
        InputLabel = np.zeros([self.batch_size, 7], dtype = np.float32)

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

            # if self.debug:
                # print "imgName: ", imgName
            img = cv2.imread(imgName)

            if img != None:                    
                img = cv2.resize(img,(self.imSize, self.imSize))
                # print "img.shape: ", img.shape
                (w, h, _) = img.shape
                x, y = ut.unpackLandmarks(labelsPTS, self.imSize)

                # newImg, newX, newY = img, x, y    

                for method in generateFunc:
                    # tag = random.choice(generateFunc)
                    if method == "resize":
                        newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
                    elif method == "rotate":
                        newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
                    elif tag == "mirror":
                        newImg, newX, newY = ut.mirror(img, x, y, w = w, h = h)
                    elif tag == "translate":
                        newImg, newX, newY = ut.translate(img, x, y, w = w, h = h)
                    elif tag == "brightnessAndContrast":
                        newImg, newX, newY = ut.contrastBrightess(img, x, y, w = w, h = h)
                    elif tag = "original":
                        newImg, newX, newY = img, x, y
                    else:
                        raise "not existing function"

                    if self.debug:
                        plotOriginal = ut.plotLandmarks(img, x, y, self.imSize, ifReturn = True)
                        plotNew = ut.plotLandmarks(newImg, newX, newY, self.imSize, ifReturn = True)

                        cv2.imwrite(self.outputDir + 'testOriginal' + str(count) + '.jpg', img)
                        cv2.imwrite(self.outputDir + 'testNew' + str(count) + '.jpg', newImg)        
                        cv2.imwrite(self.outputDir + 'plotOriginal' + str(count) + '.jpg', plotOriginal)
                        cv2.imwrite(self.outputDir + 'plotNew' + str(count) + '.jpg', plotNew)

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


                    normX = ut.normalize(newX, self.imSize)
                    normY = ut.normalize(newY, self.imSize)
                    normPTS = np.asarray(ut.packLandmarks(normX, normY))
                    normXMean, normYMean, normEdge = ut.normalize(newXMean, self.imSize), ut.normalize(newYMean, self.imSize), ut.normalize(newEdge, self.imSize)
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






    def train_on_batch(self, nb_epoch, MaxIters):
        if os.path.exists(self.modelDir)==False:
            os.mkdir(self.modelDir)
        testCount = 0
        trainCount = 0
        for e in range(nb_epoch):
            # if e>0:
            shuffle(self.DataTr)
            iterTest=0
            for iter in range (self.MaxIters):
                train_start=iter*self.batch_size
                train_end = (iter+1)*self.batch_size
                # print "train_start: ", train_start
                # print "train_end: ", train_end
                X_batch, label_BB, Z_Names = self.DataGenBB(self.DataTr, train_start=train_start, train_end=train_end)

                # print "X_batch.shape: ", X_batch.shape
                for i in range(self.batch_size):
                    labels = label_BB[i]
                    img = X_batch[i]
                    # print "input ut.deNormalize(labels): ", ut.deNormalize(labels)
                    # labelImg = ut.plotTarget(img, labels)
                    labelImg = ut.plotTarget(img, ut.deNormalize(labels, self.imSize), self.imSize)
                    cv2.imwrite(self.outputDir + 'inputTrainlabelImg' + str(trainCount) + '.jpg', labelImg)

                loss, tras, pred = self.model.train_on_batch(X_batch,label_BB)
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

                if iter%100 == 0:
                    logInfo = ""
                    if os.path.exists(self.outputDir + 'log.txt') and self.init == False:
                        f = open(self.outputDir + 'log.txt', 'a')
                    else:
                        f = open(self.outputDir + 'log.txt','w')
                        self.init = False

                    iterationInfo = ("^^^^^^^^^^^^^^^" + "\n" + 'iteration: ' + str(iter))
                    logInfo += iterationInfo
                    print iterationInfo

                    # labelImg = ut.plotTarget(X_batch[0], pred[0])
                    labelImg = ut.plotTarget(X_batch[0], ut.deNormalize(pred[0], self.imSize), self.imSize)
                    cv2.imwrite(self.outputDir + 'predTrainLabelImg' + str(trainCount) + '.jpg', labelImg)


                    test_start = iterTest * self.batch_size
                    test_end = (iterTest + 1) * self.batch_size
                    X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(self.DataTr, train_start=test_start, train_end=test_end)
                    loss, tras, pred = self.model.evaluate(X_batch_T,label_BB_T)
                    testCount += 1

                    if testCount >= 20:
                        testCount = 0

                    # labelImg = ut.plotTarget(X_batch_T[0], pred[0])
                    labelImg = ut.plotTarget(X_batch_T[0], ut.deNormalize(pred[0], self.imSize), self.imSize)
                    cv2.imwrite(self.outputDir + 'predTestLabelImg' + str(testCount) + '.jpg', labelImg)

                    testInfo = ("===================" + "\n" + "loss, TEST: " + str(loss))
                    logInfo += testInfo
                    print testInfo

                    # print "loss.shape: ", loss.shape
                    # print "pred, return on test: ", type(pred)
                    # print "pred.shape: ", pred.shape
                    # print "pred #######: ", pred
                    # print "tras, return on test: ", type(tras), tras
                    # print "tras.shape: ", tras.shape 

                    iterTest += self.batch_size
                    iterTest %= self.MaxTestIters
                    
                    f.write(logInfo)
                    f.close()

                if iter%3000==0:
                    self.model.save(self.modelDir + '/model%d.h5'%iter)

    def run(self):

        self.model = m.model(input_shape=(self.imSize, self.imSize, 3))
        
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.model.summary()
        self.train_on_batch(1, MaxIters = 20000)

        sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.model.summary()
        self.train_on_batch(1, MaxIters = 20000)

        sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.model.summary()
        self.train_on_batch(1, MaxIters = 20000)

        sgd = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.model.summary()
        self.train_on_batch(1, MaxIters = 20000)

        sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.model.summary()
        self.train_on_batch(1, MaxIters = 20000)
    
    def main(self):
        self.run()

if __name__ == '__main__':
    faceOrientPred().main()


