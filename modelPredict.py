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
        self.evaluationOutputDir = "./03212017_01_evaluation_output/"

        self.weightPath = "./03202017_01_square_model/model39000.h5"
        self.model = m.model(input_shape=(self.imSize, self.imSize, 3), weights_path = self.weightPath)
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])


    def final_pred(self, y_true, y_pred):
        # y_cont=np.concatenate(y_pred,axis=1)
        return y_pred


    def loadData(self):
        TestPath = '/home/james/CropBB15/300WBB15challengeTest.txt'

        FTe = open(TestPath,'r')
        self.DataTe = FTe.readlines()
        TeNum = len(self.DataTe)

        print "len(self.DataTe): ", len(self.DataTe)

        self.MaxTestIters = TeNum/self.batch_size
        print "test data length:", TeNum
        print "self.MaxTestIters: ", self.MaxTestIters


    def DataGenBB(self, DataStrs, train_start,train_end):
        generateFunc = ["original", "resize", "rotate", "brightnessAndContrast" ]
        # generateFunc = ["original", "resize", "rotate", "mirror", "translate", "brightnessAndContrast" ]

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
            labelsPTS=labels[:136].reshape([68,2])
            img = cv2.imread(imgName)

            if img != None:   
                img = cv2.resize(img,(self.imSize, self.imSize))
                (w, h, _) = img.shape
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


                    InputData[count,...] = newImg
                    labels = np.array([normXMean, normYMean, normEdge])
                    InputLabel[count,...] = labels
                    InputNames.append(imgName)

                    count += 1

            else:
                print "cannot : ", imgName


        return InputData, InputLabel, np.asarray(InputNames)

    def predcit(self):
        for iter in range (self.MaxTestIters):
            test_start = iter * self.batch_size
            test_end = (iter + 1) * self.batch_size
            X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(self.DataTe, train_start=test_start, train_end=test_end)
            # loss, tras, pred = self.model.evaluate(X_batch_T,label_BB_T)
            print "X_batch_T.shape: ", X_batch_T.shape
            print "label_BB_T.shape: ", label_BB_T.shape

            pred = self.model.predict(X_batch_T, batch_size = 32, verbose=1)
            print "type(pred): ", type(pred)
            print "pred.shape: ", pred.shape


            for i in range(self.batch_size):
                labels = label_BB_T[i]
                img = X_batch_T[i]

                labelImg = ut.plotTarget(img, ut.deNormalize(labels, self.imSize), self.imSize, ifSquareOnly = True)
                cv2.imwrite(self.evaluationOutputDir + 'inputTestImg' + str(iter * self.batch_size + i) + '.jpg', img)
                labelImg = ut.plotTarget(img, ut.deNormalize(pred[i], self.imSize), self.imSize, ifSquareOnly = True)
                print 'save predTestLabelImg' + str(iter * self.batch_size + i) + '.jpg to: ' + self.outputDir
                cv2.imwrite(self.evaluationOutputDir + 'predTestLabelImg' + str(iter * self.batch_size + i) + '.jpg', labelImg)


    def run(self):
        pass
        self.loadData()
        self.predcit()

if __name__ == '__main__':
    ModelPredict().run()


