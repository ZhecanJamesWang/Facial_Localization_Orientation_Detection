# -*- coding: utf-8 -*-
import utility as ut
import numpy as np
import cv2
import os
import random

class filterBoundingBox(object):
	def __init__(self):
		self.rawDir = "competitionImageDataset/testset/semifrontal/"
		self.files = os.listdir(self.rawDir)
		self.outputDir = "competitionImageDataset/filterOutput03252017_02_ratio_100202/"
		# print len(self.files)
		self.centerWeight = 10
		self.sizeWeight = 2
		self.confidenceWeight = 2
		self.debug = False

	def plotLandmarks(image, X, Y, imSize = None, name = None, ifRescale = False, ifReturn = False, circleSize = 2):
	    # plot landmarks on original image
	    img = np.copy(image)
	    assert len(X) == len(Y)   
	    for index in range(len(X)):
	        if ifRescale:
	            (w, h, _) = img.shape
	            cv2.circle(img,(int((X[index] + 0.5) * imSize), int((Y[index] + 0.5) * imSize)), circleSize, (0,0,255), -1)
	        else:
	            cv2.circle(img,(int(X[index]), int(Y[index])), circleSize, (0,0,255), -1)
	    if ifReturn:
	        return img
	    else:
	        cv2.imshow(name,img)

	def plotTarget(self, image, labels, ifGreen = False):
	    img = np.copy(image)
	    if ifGreen:
	        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 255, 0), 3)
	    else:
	        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 0, 255), 3)
	    return img

	def run(self):
		counter = 0
		for fileName in self.files:
			if ".JSBB" in fileName:

				fileHeader = fileName.split(".")[0]
				imgName = fileHeader + ".jpg"
				img = cv2.imread(self.rawDir + imgName)
				w, h, _ = img.shape
				f = open(self.rawDir + fileName, 'r+')
				lines = f.readlines()
				confidenceList, edgeList, middleRatioList, scoreList = [], [], [], []
				content = []
				# rankList = [0] * len(lines)
				if self.debug:
					print "len(lines): ", len(lines)
				if len(lines) == 0:
					print "len(lines) == 0: fileHeader: ", fileHeader
				boxNum = len(lines)
				for line in lines:
					if self.debug:
						print "line: ", line
					try:
						confidence, xMin, yMin, xMax, yMax = line.split(" ")
						confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax)

					except Exception as e:
						confidence, xMin, yMin, xMax, yMax = line.split("\t")
						confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax[:-1])
						# raise "debug"
					if self.debug:
						plotImg = self.plotTarget(img, [xMin, yMin, xMax, yMax])
						cv2.imwrite(self.outputDir + fileHeader + "_bb0_" + str(boxNum) + '_inputFirstBox.jpg', plotImg)

					content.append([xMin, yMin, xMax, yMax])
					confidenceList.append(confidence)
					edgeList.append(max(xMax - xMin, yMax - yMin))
				

					xMean = (xMin + xMax)/2.0
					yMean = (yMin + yMax)/2.0

					value = (abs(xMean - w/2.0) + abs(yMean - h/2.0))/2.0
					if value == 0:
						middleRatioList.append(1)
					else:
						middleRatioList.append(value)
				
				for index in range(len(content)):
					score = confidenceList[index] * self.confidenceWeight 
					score += edgeList[index] * self.sizeWeight
					try:
						score = score/float(middleRatioList[index] * self.centerWeight)
					except Exception as e:
						print e
						print middleRatioList[index]
						raise "debug"
					scoreList.append(score)

				# confidenceList, edgeList, middleRatioList, scoreList = np.asarray(confidenceList), np.asarray(edgeList), np.asarray(middleRatioList), np.asarray(scoreList)			

				# confidenceRank = np.argmax(confidenceList)
				# rankList[confidenceRank] += 1
				# edgeRank = np.argmax(edgeList)
				# rankList[edgeRank] += 1
				# middleRatioRank = np.argmax(middleRatioList)
				# rankList[middleRatioRank] += 1


	
				finalContent = content[np.argmax(scoreList)]
				if self.debug:
					print "finalContent: ", finalContent
				xMin, yMin, xMax, yMax = finalContent	
				xMean = (xMax - xMin)/2.0
				yMean = (yMax - yMin)/2.0
				edge = max()

				f = open(self.outputDir + fileHeader + "_bb0" + str(boxNum) + '.JSBB','w') 
				f.write(str(finalContent[0]) + " " + str(finalContent[1]) + " " + str(finalContent[2]) +  " " + str(finalContent[3]))
				plotImg = self.plotTarget(img, finalContent, ifGreen = True)
				cv2.imwrite(self.outputDir + fileHeader + "_bb0" + str(boxNum) + '.jpg', plotImg)

				counter += 1
				if counter % 100 == 0:
					print "counter: ", counter


if __name__ == '__main__':
	filterBoundingBox().run()


