# -*- coding: utf-8 -*-
from keras import optimizers
import utility as ut
import numpy as np
import selfModel as m
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ModelPredict(object):
    def __init__(self):
        self.batch_size = 32
        self.imSize = 256
        self.evaluationOutputDir = "./03222017_02_Menpo39_evaluation_output/"

        self.weightPath = "./03202017_01_square_model/model39000.h5"
        self.model = m.model(input_shape=(self.imSize, self.imSize, 3), weights_path = self.weightPath)
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        self.ifMenpo39DataSet = True


    def final_pred(self, y_true, y_pred):
        # y_cont=np.concatenate(y_pred,axis=1)
        return y_pred


    def loadData(self):
        if self.ifMenpo39DataSet:
            self.ImgDir = "./Menpo39Preprocessed/img/"
            self.PTSDir = "./Menpo39Preprocessed/pts/"
            self.imgs = os.listdir(self.ImgDir)
            TeNum = len(self.imgs)
        else:
            TestPath = '/home/james/CropBB15/300WBB15challengeTest.txt'
            FTe = open(TestPath,'r')
            self.DataTe = FTe.readlines()
            TeNum = len(self.DataTe)
            print "len(self.DataTe): ", len(self.DataTe)

        self.MaxTestIters = TeNum/self.batch_size
        print "test data length:", TeNum
        print "self.MaxTestIters: ", self.MaxTestIters

    def unpackLandmarks(self, array):
        x = []
        y = []
        for i in range(0, len(array)):
            x.append(array[i][0])
            y.append(array[i][1])
        return x, y

    def DataGenBB(self, train_start,train_end, DataStrs = None):
        generateFunc = ["original"]
        # generateFunc = ["original", "resize", "rotate", "mirror", "translate", "brightnessAndContrast" ]

        InputData = np.zeros([self.batch_size * len(generateFunc), self.imSize, self.imSize, 3], dtype = np.float32)
        # InputLabel = np.zeros([self.batch_size * len(generateFunc), 7], dtype = np.float32)
        InputLabel = np.zeros([self.batch_size * len(generateFunc), 3], dtype = np.float32)

        InputNames = []
        count = 0
        for i in range(train_start,train_end):
            if self.ifMenpo39DataSet:
                imgName =self.imgs[i]
                imgNameHeader = imgName.split('.')[0]
                index = imgNameHeader[imgNameHeader.find('e') + 1:]
                labelsPTS = np.loadtxt(self.PTSDir + 'pts' + index + ".txt")
                img = cv2.imread(self.ImgDir + imgName)
            else:
                strLine = DataStrs[i]
                strCells = strLine.rstrip(' \n').split(' ')
                imgName = strCells[0]

                labels = np.array(strCells[1:]).astype(np.float)
                labelsPTS=labels[:136].reshape([68,2])
                img = cv2.imread(imgName)



            if img != None:  
                print "img.shape: ", img.shape

                img = cv2.resize(img,(self.imSize, self.imSize))

                (w, h, _) = img.shape
                if not self.ifMenpo39DataSet:
                    # x, y = self.unpackLandmarks(labelsPTS)
                    x, y = None, None
                else:
                    x, y = ut.unpackLandmarks(labelsPTS, self.imSize)

                for index in range(len(generateFunc)):
                    method = generateFunc[index]
                    # tag = random.choice(generateFunc)
                    if method == "resize":
                        newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
                    elif method == "rotate":
                        newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
                    elif method == "mirror":
                        newImg, newX, newY = ut.mirror(img, x, y, w = w, h = h)
                    elif method == "translate":
                        newImg, newX, newY = ut.translate(img, x, y, w = w, h = h)
                    elif method == "brightnessAndContrast":
                        newImg, newX, newY = ut.contrastBrightess(img, x, y)
                    elif method == "original":
                        newImg, newX, newY = img, x, y
                    else:
                        raise "not existing function"
                if self.ifMenpo39DataSet:
                    labels = labelsPTS
                else:
                    newXMin = min(newX)
                    newYMin = min(newY)
                    newXMax = max(newX)
                    newYMax = max(newY)
                    newXMean = (newXMax + newXMin)/2.0
                    newYMean = (newYMax + newYMin)/2.0
                    newEdge = max(newYMax - newYMin, newXMax - newXMin)
                                

                    normX = ut.normalize(newX, self.imSize)
                    normY = ut.normalize(newY, self.imSize)
                    normXMean, normYMean, normEdge = ut.normalize(newXMean, self.imSize), ut.normalize(newYMean, self.imSize), ut.normalize(newEdge, self.imSize)
                    labels = np.array([normXMean, normYMean, normEdge])

                InputData[count,...] = newImg
                InputLabel[count,...] = labels
                InputNames.append(imgName)

                count += 1

            else:
                print "cannot : ", imgName


        return InputData, InputLabel, np.asarray(InputNames)

    def predcit(self):
        saveCount = 0
        for iter in range (self.MaxTestIters):
            test_start = iter * self.batch_size
            test_end = (iter + 1) * self.batch_size
            # if iter == self.MaxTestIters - 1:
            #     test_end = len(self.DataTe)
            if self.ifMenpo39DataSet:
                X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(train_start = test_start, train_end = test_end)
            else:
                X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(DataStrs = self.DataTe, train_start = test_start, train_end = test_end)
           
            print "X_batch_T.shape: ", X_batch_T.shape
            print "label_BB_T.shape: ", label_BB_T.shape

            pred = self.model.predict(X_batch_T, verbose=1)
            print "type(pred): ", type(pred)
            print "pred.shape: ", pred.shape


            for i in range(self.batch_size):
                labels = label_BB_T[i]
                img = X_batch_T[i]

                img = ut.plotTarget(img, ut.deNormalize(labels, self.imSize), self.imSize, ifSquareOnly = True,  ifGreen = True)
                cv2.imwrite(self.evaluationOutputDir + 'inputTestImg' + str(saveCount) + '.jpg', img)
                labelImg = ut.plotTarget(img, ut.deNormalize(pred[i], self.imSize), self.imSize, ifSquareOnly = True)
                print 'save predTestLabelImg' + str(saveCount) + '.jpg to: ' + self.evaluationOutputDir
                cv2.imwrite(self.evaluationOutputDir + 'predTestLabelImg' + str(saveCount) + '.jpg', labelImg)
                saveCount += 1



    def run(self):
        pass
        self.loadData()
        self.predcit()

if __name__ == '__main__':
    ModelPredict().run()


