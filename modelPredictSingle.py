# -*- coding: utf-8 -*-
from keras import optimizers
import utility as ut
import numpy as np
import selfModel as m
import cv2
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ModelPredict(object):
    def __init__(self):
        self.batch_size = 32
        self.imSize = 256

        self.weightPath = "./transferTmp/03202017_02_square_add_layers_model/model39000.h5"
        self.model = m.model(input_shape=(self.imSize, self.imSize, 3), weights_path = self.weightPath)
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', self.final_pred])
        

        self.evaluationOutputDir = "./output/03292017_01_preProcessedSemifrontal_bigNet_output/"
        self.ImgDir= 'data/preProcessedSemifrontal/img/'
        self.labelDir= 'data/preProcessedSemifrontal/label/'
        # self.evaluationOutputDir = "./output/03292017_02_preProcessedProfile_bigNet_output/"
        # self.ImgDir= 'data/preProcessedProfile/img/'
        # self.labelDir= 'data/preProcessedProfile/label/'

        self.imgs = os.listdir(self.ImgDir)

        self.debug = False

    def final_pred(self, y_true, y_pred):
        # y_cont=np.concatenate(y_pred,axis=1)
        return y_pred


    def loop(self):
        for i in range(len(self.imgs)):
            imgName =self.imgs[i]
            # print "imgName: ", imgName
            fileHeader = imgName.split('.')[0]
            print "self.ImgDir + imgName: ", self.ImgDir + imgName
            img = cv2.imread(self.ImgDir + imgName)
            print "self.labelDir + imgNameHeader + .txt: ", self.labelDir + fileHeader + ".txt"
            label = np.loadtxt(self.labelDir + fileHeader + ".txt")
    
            if self.debug:
                img = ut.plotTarget(img, label, ifSquareOnly = True, ifGreen = True)
                cv2.imshow("img", img)
                cv2.waitKey(0) 

            if img != None:  
            #     print "img.shape: ", img.shape
                w, h, _ = img.shape
                xMean, yMean, edge = label
                xMean = xMean/w
                yMean = yMean/h
                edge = edge/w
                img = cv2.resize(img,(self.imSize, self.imSize))
                xMean = xMean*self.imSize
                yMean = yMean*self.imSize
                edge = edge*self.imSize   
                label = np.asarray([xMean, yMean, edge])
            else:
                print "cannot find : ", imgName

            InputData = np.zeros([1, self.imSize, self.imSize, 3], dtype = np.float32)
            InputLabel = np.zeros([1, 3], dtype = np.float32)
            InputData[0] = img
            self.predcit(InputData, label, fileHeader)

    def predcit(self, img, label, index):
        if self.debug:
            cv2.circle(img[0],(int(label[0]), int(label[1])), 3, (255, 0, 0), -1)

        print "img.shape: ", img.shape
        print "label.shape: ", label.shape

        pred = self.model.predict(img, verbose=1)    
        print "pred.shape: ", pred.shape

        img = ut.plotTarget(img[0], label, ifSquareOnly = True,  ifGreen = True)
        # img = ut.plotTarget(img, ut.deNormalize(labels, self.imSize), self.imSize, ifSquareOnly = True,  ifGreen = True)
        # cv2.imwrite(self.evaluationOutputDir + 'inputTestImg' + str(saveCount) + '.jpg', img)
        labelImg = ut.plotTarget(img, ut.deNormalize(pred[0], self.imSize), self.imSize, ifSquareOnly = True)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        print "saving" + self.evaluationOutputDir + str(index) + 'Pred' + '.jpg'
        # cv2.imwrite(self.evaluationOutputDir + str(index) + 'Input' + '.jpg', img)
        cv2.imwrite(self.evaluationOutputDir + str(index) + 'Pred' + '.jpg', labelImg)



    def run(self):
        self.loop()

if __name__ == '__main__':
    ModelPredict().run()


