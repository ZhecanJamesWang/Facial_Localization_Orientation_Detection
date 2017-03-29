# -*- coding: utf-8 -*-
import utility as ut
import numpy as np
import cv2
import os
import random
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreProcessMenpo39(object):
	def __init__(self):
		self.rawDataDir = "/home/james/Menpo39_Valid/"
		self.filterImgDir = "./Menpo39Preprocessed128/img/"
		self.filterPTSDir = "./Menpo39Preprocessed128/pts/"		
		self.debug = False
		self.imSize = 128

	def getDataByFiles(self):
		counter = 0
		files = os.listdir(self.rawDataDir)
		print "len(files): ", len(files)
		for file in files:
			# if file != ".DS_Store" in file:
			if ".pts39" in file:
				fileHeader = file.split(".")[0]
				print "fileHeader: ", fileHeader
				img = cv2.imread(self.rawDataDir + fileHeader + ".jpg")
				print "img.shape: ", img.shape
				print "file name: ", file
				pts = np.loadtxt(self.rawDataDir + file)
				print "type(pts): ", type(pts)
				print "pts.shape: ", pts.shape
				x, y = self.unpackLandmarks(pts)
				# if self.debug:
				# img = ut.plotLandmarks(img, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				# cv2.imwrite(self.filterImgDir + 'orginalImage' + str(counter) + '.jpg', img)
				cropImg, labels = self.process(img, x, y)
				# cv2.imwrite('testRectangle' + str(counter)  + '.jpg', testImg)
				# print "cropImg.shape: ", cropImg.shape
				# print "pts.shape: ", pts.shape
				# print "str(counter): ", str(counter)
				# cv2.imwrite(self.filterImgDir + 'CroppedImage' + str(counter) + '.jpg', cropImg)
				# if self.debug:
				# img = ut.plotLandmarks(cropImg, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				# cv2.imwrite(self.filterImgDir + 'testCropImgLandmarks' + str(counter) + '.jpg', img)	

				w, h, _ = cropImg.shape
				# x = x / w
				# y = y / h
				# x = x * 256
				# y = y * 256
				labels[0] = labels[0] / w
				labels[1] = labels[1] / h
				labels[2] = labels[2] / w
				labels[0] = labels[0] * 256
				labels[1] = labels[1] * 256
				labels[2] = labels[2] * 256

				resizeImg = cv2.resize(cropImg,(self.imSize, self.imSize))
				# resizeImg, x, y = ut.resize(cropImg, x, y, size = (256, 256))
				# cv2.imwrite(self.filterImgDir + 'ResizedImage' + str(counter) + '.jpg', resizeImg)

				# img = ut.plotLandmarks(resizeImg, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				# cv2.imwrite(self.filterImgDir + 'testResizeImgLandmarks' + str(counter) + '.jpg', img)


				# pts = np.asarray(ut.packLandmarks(x, y))


				# np.savetxt(self.filterPTSDir + 'pts' + str(counter) + '.txt', pts)
				np.savetxt(self.filterPTSDir + 'pts' + str(counter) + '.txt', labels)
				cv2.imwrite(self.filterImgDir+ 'image' + str(counter) + '.jpg', resizeImg)
				counter += 1


	def process(self, img, x, y):
		w, h, _ = img.shape
		xMin = min(x)
		yMin = min(y)
		xMax = max(x)
		yMax = max(y)
		xMean = (xMax + xMin)/2.0
		yMean = (yMax + yMin)/2.0
		edge = max(yMax - yMin, xMax - xMin)
		# ground-truth center, W -> (center*disturbance（+-10%）, W*1.5*disturbance（+-20%）)  --》 new center, W
		# newXMean = xMean * self.getDisturbance(0.1)
		# newYMean = yMean * self.getDisturbance(0.1)
		# newEdge = edge * 1.5 * self.getDisturbance(0.2)
		newXMean = xMean 
		newYMean = yMean 
		newEdge = edge * 1.5


		# if self.debug:
		labels = np.array([xMean, yMean, edge])
		# img = ut.plotTarget(img, labels, ifSquareOnly = True)
		# labels = [newXMean, newYMean, newEdge]
		testImg = None
		# testImg = ut.plotTarget(img, labels, ifSquareOnly = True, ifGreen = True)
		# cv2.imwrite('testRectangle.jpg', img)
		edgeList = [newEdge]
///////////////////////////////////////////////////////////////
		if int(newXMean - newEdge/2.0)  < 0:
			# newXMin = 0  
			edgeList.append(2 * int(newXMean - 0))
		# else:
			# newXMin = int(newXMean - newEdge/2.0) 

		if int(newXMean + newEdge/2.0) > w :
			# newXMax = xMax  
			edgeList.append(2 * int(w - newXMean)) 
		# else:
		# 	newXMax = int(newXMean + newEdge/2.0)

		if int(newYMean - newEdge/2.0)  < 0:
			# newYMin = 0  
			edgeList.append(2 * int(newYMean - 0)) 
		# else: 
		# 	newYMin = int(newYMean - newEdge/2.0) 
		
		if int(newYMean + newEdge/2.0) > h:
			# newYMax = yMax 
			edgeList.append(2 * int(h - newYMean))  
		# else:
		# 	newYMax = int(newYMean + newEdge/2.0)
///////////////////////////////////////////////////////////////

		newEdge = min(edgeList)
		# cropImg = img[ int(newYMin) : int(newYMax), int(newXMin) : int(newXMax)]
		print "img.shape: ", img.shape
		print "newXMean: ", newXMean
		print "newYMean: ", newYMean
		print "newEdge: ", newEdge
		print int(newYMean - newEdge/2.0)
		print int(newYMean + newEdge/2.0)
		print int(newYMean - newEdge/2.0) 
		print int(newXMean + newEdge/2.0)
		cropImg = img[int(newYMean - newEdge/2.0) : int(newYMean + newEdge/2.0), int(newXMean - newEdge/2.0) : int(newXMean + newEdge/2.0)]
		print "cropImg.shape: ", cropImg.shape

		x = np.asarray(x)
		y = np.asarray(y)
		x = x - int(newXMean - newEdge/2.0)
		y = y - int(newYMean - newEdge/2.0)	
		return cropImg, labels


	def getDisturbance(self, value):
		return 1 + random.uniform(-value, value)
		

	def run(self):
		self.getDataByFiles()

	def unpackLandmarks(self, array):
		x = []
		y = []
		for i in range(0, len(array)):
			x.append(array[i][0])
			y.append(array[i][1])
		return x, y



if __name__ == '__main__':
	PreProcessMenpo39().run()