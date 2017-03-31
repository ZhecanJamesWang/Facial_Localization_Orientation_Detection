import cv2
import numpy as np
import os
from random import shuffle
# from PIL import Image
# from BK.MeshNetLayer import *
#from Lib3D.MeshModel import *
#from Lib3D.Rasterization import  *
#from Lib3D.Util2D import *
#from Lib3D.Util3D import *
from keras import optimizers
# from keras.callbacks import ModelCheckpoint
from keras import backend as K
import utility as ut
import random
# from keras import callbacks
import shutil
# import model as m
import selfModel as m
from decimal import Decimal
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="1"




class faceOrientPred(object):
    """face orientation detection"""
    def __init__(self):

        self.init = True
        self.debug = False
        self.outputDir = "./output/03312017_01_bigNet_Scale_Menpo39_output/"
        self.modelDir = "./output/03312017_01_bigNet_Scale_Menpo39_model/"
        self.imSize = 256

        # TN = TextNet('./MatBS/shape_0.obj', imgW=256)
        # TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'    
        TrainPath = "/home/james/Facial_Localization_Orientation_Detection/data/combineData.txt"
        TestPath = '/home/james/CropBB15/300WBB15challengeTest.txt'
        # TestPath = "data/Menpo39TrainProcessed/menpo39Data.txt"

        FTr = open(TrainPath,'r')
        self.DataTr = FTr.readlines()
        TrNum = len(self.DataTr)

        # self.DataTe = self.DataTr[:int(len(self.DataTr)*0.1)]
        # self.DataTr = self.DataTr[int(len(self.DataTr)*0.1):]
        # self.DataTe = self.DataTr

        FTe = open(TestPath,'r')
        self.DataTe = FTe.readlines()
        TeNum = len(self.DataTe)

        print "len(self.DataTr): ", len(self.DataTr)
        print "len(self.DataTe): ", len(self.DataTe)


        shuffle(self.DataTr)

        # print "type(DataTr): ", type(self.DataTr)
        # print "len(DataTr): ", len(self.DataTr)

        self.batch_size = 32
        self.MaxIters = TrNum/self.batch_size
        self.MaxTestIters = TeNum/self.batch_size

        print "train data length:", TrNum
        print "test data length:", TeNum
        print "self.MaxIters: ", self.MaxIters
        print "self.MaxTestIters: ", self.MaxTestIters

        self.ifMenpo39Data = False

    def final_pred(self, y_true, y_pred):
        # y_cont=np.concatenate(y_pred,axis=1)
        return y_pred

    def unpackLandmarks(self, array):
        x = []
        y = []
        for i in range(0, len(array)):
            x.append(array[i][0])
            y.append(array[i][1])
        return x, y

    def DataGenBB(self, DataStrs, train_start,train_end):
        generateFunc = ["original", "scale", "rotate", "translate", "scaleAndTranslate", "brightnessAndContrast"]
        # generateFunc = ["original", "scale", "rotate", "translate", "scaleAndTranslate"]

        InputData = np.zeros([self.batch_size * len(generateFunc), self.imSize, self.imSize, 3], dtype = np.float32)
        # InputLabel = np.zeros([self.batch_size * len(generateFunc), 7], dtype = np.float32)
        InputLabel = np.zeros([self.batch_size * len(generateFunc), 3], dtype = np.float32)

        InputNames = []
        count = 0
        for i in range(train_start,train_end):
            strLine = DataStrs[i]
            strCells = strLine.rstrip(' \n').split(' ')
            imgName = strCells[0]

            labels = np.array(strCells[1:]).astype(np.float)

            if len(labels) == 78:
                # print "switch to menpo39"
                labelsPTS=labels[:136].reshape([39,2])
                self.ifMenpo39Data = True                
            else:            
                # print "not menpo39"
                labelsPTS=labels[:136].reshape([68,2])
                self.ifMenpo39Data = False

            # if self.debug:
            #     print "imgName: ", imgName
            img = cv2.imread(imgName)

            if img != None:   
                # print "find image: ", imgName  
                # print "img.shape: ", img.shape

                img = cv2.resize(img,(self.imSize, self.imSize))
                # print "img.shape: ", img.shape
                if self.ifMenpo39Data:
                    x, y = self.unpackLandmarks(labelsPTS)
                else:
                    x, y = ut.unpackLandmarks(labelsPTS, self.imSize)

                # newImg, newX, newY = img, x, y    

                for index in range(len(generateFunc)):
                    method = generateFunc[index]
                    (w, h, _) = img.shape
                    # tag = random.choice(generateFunc)
                    if method == "resize":
                        newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
                    elif method == "rotate":
                        newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
                    elif method == "mirror":
                        newImg, newX, newY = ut.mirror(img, x, y, w = w, h = h)
                    elif method == "translate" or method == "scaleAndTranslate": 
                        newImg, newX, newY = ut.translate(img, x, y, w = w, h = h)
                    elif method == "brightnessAndContrast":
                        newImg, newX, newY = ut.contrastBrightess(img, x, y)
                    elif method == "original":
                        newImg, newX, newY = img, x, y
                    elif method == "scale":
                        newImg, newX, newY = img, x, y
                    else:
                        raise "not existing function"

                    # if self.debug:
                    #     plotOriginal = ut.plotLandmarks(img, x, y, self.imSize, ifReturn = True)
                    #     plotNew = ut.plotLandmarks(newImg, newX, newY, self.imSize, ifReturn = True)

                    #     cv2.imwrite(self.outputDir + 'testOriginal' + str(count) + '.jpg', img)
                    #     cv2.imwrite(self.outputDir + 'testNew' + str(count) + '.jpg', newImg)        
                    #     cv2.imwrite(self.outputDir + 'plotOriginal' + str(count) + '.jpg', plotOriginal)
                    #     cv2.imwrite(self.outputDir + 'plotNew' + str(count) + '.jpg', plotNew)

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
                    edge = max(newYMax - newYMin, newXMax - newXMin)
                    
                    # if method == "scale":
                    #     cv2.imshow("originalImg", newImg)
                    #     cv2.waitKey(0)

                    if method == "scale" or method == "scaleAndTranslate":
                        newEdge = np.random.uniform(0.7, 0.9) * edge
                        newXMin = int(newXMean - newEdge/2.0)
                        newXMax = int(newXMean + newEdge/2.0)
                        newYMin = int(newYMean - newEdge/2.0)
                        newYMax = int(newYMean + newEdge/2.0)
                        
                        newXMean = newXMean - newXMin
                        newYMean = newYMean - newYMin
                        
                        # print "newXMin, newYMin, newXMax, newYMax: ", newXMin, newYMin, newXMax, newYMax
                        
                        newImg = Image.fromarray(newImg.astype(np.uint8))
                        cropImg = newImg.crop((newXMin, newYMin, newXMax, newYMax))
                        newImg = np.array(cropImg)

                        # cv2.imshow("processing", newImg)
                        # cv2.waitKey(0)

                        w, h, _ = newImg.shape
                        edge = edge*self.imSize/w
                        newXMean = newXMean*self.imSize/w
                        newYMean = newYMean*self.imSize/h
                        newImg = cv2.resize(newImg,(self.imSize, self.imSize))

                    # print "newXMin: ", newXMin
                    # print "newYMin: ", newYMin
                    # print "newXMax: ", newXMax
                    # print "newYMax: ", newYMax
                    # print "newXMean: ", newXMean
                    # print "newYMean: ", newYMean
                    # print "newEdge: ", newEdge
                    
                    # if method == "scale":
                    #     newImg = ut.plotTarget(newImg, [newXMean, newYMean, edge], ifSquareOnly = True, ifGreen = True)
                    #     cv2.imshow("newImg", newImg)
                    #     cv2.waitKey(0)

                    # if self.ifMenpo39Data == False:
                    #     cv2.imwrite(str(count) + str(method) + '.jpg', newImg)


                    normX = ut.normalize(newX, self.imSize)
                    normY = ut.normalize(newY, self.imSize)
                    # normPTS = np.asarray(ut.packLandmarks(normX, normY))
                    normXMean, normYMean, normEdge = ut.normalize(newXMean, self.imSize), ut.normalize(newYMean, self.imSize), ut.normalize(edge, self.imSize)
                    # print "newPTS: ", newPTS.shape

                    # print "ut.deNormalize(normXMin): ", ut.deNormalize(normXMin)
                    # print "ut.deNormalize(normYMin): ", ut.deNormalize(normYMin)
                    # print "ut.deNormalize(normXMax): ", ut.deNormalize(normXMax)
                    # print "ut.deNormalize(normYMax): ", ut.deNormalize(normYMax)
                    # print "ut.deNormalize(normXMean): ",ut.deNormalize(normXMean)
                    # print "ut.deNormalize(normYMean): ",ut.deNormalize(normYMean)
                    # print "ut.deNormalize(normEdge): ", ut.deNormalize(normEdge)
                    
                    # print "method: ", method
                    # print "newImg.shape: ", newImg.shape

                    # print "len(InputData): ", len(InputData)
                    InputData[count,...] = newImg
                    # labels = np.array([normPTS[27][0], normPTS[27][1], normPTS[8][0], 
                    #     normPTS[8][1], normXMean, normYMean, normEdge])
                    labels = np.array([normXMean, normYMean, normEdge])
                    InputLabel[count,...] = labels
                    InputNames.append(imgName)

                    # print "count: ", count
                    count += 1

            else:
                print "cannot : ", imgName


        return InputData, InputLabel, np.asarray(InputNames)


    # def reset_model(self, model):
    #     for layer in model.layers:
    #         if hasattr(layer, 'init'):
    #             init = getattr(layer, 'init')
    #             new_weights = init(layer.get_weights()[0].shape).get_value()
    #             bias = shared_zeros(layer.get_weights()[1].shape).get_value()
    #             layer.set_weights([new_weights, bias])

    def reset_weights(self, model):
        session = K.get_session()
        for layer in model.layers:
            if isinstance(layer, Dense):
                old = layer.get_weights()
                layer.W.initializer.run(session=session)
                layer.b.initializer.run(session=session)
                print(np.array_equal(old, layer.get_weights())," after initializer run")
            else:
                print(layer, "not reinitialized")

    def train_on_batch(self, nb_epoch, MaxIters):
        if os.path.exists(self.modelDir)==False:
            os.mkdir(self.modelDir)
        testCount = 0
        trainCount = 0
        for e in range(nb_epoch):

            shuffle(self.DataTr)
            iterTest=0
            for iter in range (self.MaxIters):
                train_start=iter*self.batch_size
                train_end = (iter+1)*self.batch_size
                X_batch, label_BB, Z_Names = self.DataGenBB(self.DataTr, train_start=train_start, train_end=train_end)

                # for i in range(self.batch_size):
                #     labels = label_BB[i]
                #     img = X_batch[i]
                    # labelImg = ut.plotTarget(img, ut.deNormalize(labels, self.imSize), self.imSize, ifSquareOnly = True)
                    # cv2.imwrite(self.outputDir + 'inputTrainlabelImg' + str(trainCount) + '.jpg', labelImg)

                loss, tras, pred = self.model.train_on_batch(X_batch,label_BB)
                trainCount += 1

                if trainCount >= 20:
                    trainCount = 0

                # if loss in [None, float("inf"), float("-inf"), Decimal('Infinity')] or "nan" in str(loss):
                #     print "-----model reset weights-----"
                #     self.reset_weights(self.model)
                #     self.reset_model(self.model)
                #     self.model.reset_states()


                # print "*****"
                print "loss, train: ", loss


                if iter%1000 == 0:
                    logInfo = ""
                    if os.path.exists(self.outputDir + 'log.txt') and self.init == False:
                        f = open(self.outputDir + 'log.txt', 'a')
                    else:
                        f = open(self.outputDir + 'log.txt','w')
                        self.init = False

                    iterationInfo = ("^^^^^" + "\n" + 'iteration: ' + str(iter))
                    logInfo += iterationInfo
                    print iterationInfo

                    # labelImg = ut.plotTarget(X_batch[0], pred[0])
                    index = random.randint(0, self.batch_size - 1)
                    labelImg = ut.plotTarget(X_batch[index], ut.deNormalize(pred[index], self.imSize), self.imSize, ifSquareOnly = True)
                    print 'save predTrainLabelImg' + str(testCount) + '.jpg to: ' + self.outputDir
                    cv2.imwrite(self.outputDir + 'predTrainLabelImg' + str(testCount) + '.jpg', labelImg)


                    test_start = iterTest * self.batch_size
                    test_end = (iterTest + 1) * self.batch_size
                    X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(self.DataTe, train_start=test_start, train_end=test_end)
                    loss, tras, pred = self.model.evaluate(X_batch_T,label_BB_T)
                    
                    testCount += 1

                    if testCount >= 20:
                        testCount = 0

                    index = random.randint(0, self.batch_size - 1)
                    labelImg = ut.plotTarget(X_batch_T[index], ut.deNormalize(pred[index], self.imSize), self.imSize, ifSquareOnly = True)
                    print 'save predTestLabelImg' + str(testCount) + '.jpg to: ' + self.outputDir
                    cv2.imwrite(self.outputDir + 'predTestLabelImg' + str(testCount) + '.jpg', labelImg)

                    testInfo = ("====" + "\n" + "loss, TEST: " + str(loss))
                    logInfo += testInfo
                    print testInfo


                    iterTest += self.batch_size
                    iterTest %= self.MaxTestIters
                    
                    f.write(logInfo)
                    f.close()

                if iter%3000==0:
                    self.model.save(self.modelDir + '/model%d.h5'%iter)

    def run(self):

        self.model = m.model(input_shape=(self.imSize, self.imSize, 3))
        
        # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        # self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        # self.model.summary()
        # self.train_on_batch(1, MaxIters = 20000)

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


