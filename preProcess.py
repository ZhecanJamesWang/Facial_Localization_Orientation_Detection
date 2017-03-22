# -*- coding: utf-8 -*-
import utility as ut
import numpy as np
import cv2
import os
import random
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreProcess(object):
	def __init__(self):
		self.rawDataDir = "/home/james/Menpo39_Valid/"
		self.filterImgDir = "./Menpo39Preprocessed/img/"
		self.filterPTSDir = "./Menpo39Preprocessed/pts/"		
		self.debug = False

	def getDataByFiles(self):
		counter = 0
		files = os.listdir(self.rawDataDir)
		print "len(files): ", len(files)
		for file in files:
			# if file != ".DS_Store" in file:
			if ".pts39" in file:
				fileHeader = file.split(".")[0]
				img = cv2.imread(self.rawDataDir + fileHeader + ".jpg")
				print "file name: ", file
				pts = np.loadtxt(self.rawDataDir + file)
				print "type(pts): ", type(pts)
				print "pts.shape: ", pts.shape
				x, y = self.unpackLandmarks(pts)
				if self.debug:
					img = ut.plotLandmarks(img, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
					cv2.imwrite('test.jpg', img)
				cropImg, x, y = self.process(img, x, y)
				print "cropImg.shape: ", cropImg.shape
				print "pts.shape: ", pts.shape
				cv2.imwrite(self.filterImgDir + 'CroppedImage' + str(counter) + '.jpg', cropImg)
				# if self.debug:
				img = ut.plotLandmarks(cropImg, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				cv2.imwrite(self.filterImgDir + 'testCropImgLandmarks' + str(counter) + '.jpg', img)	

				resizeImg, x, y = ut.resize(cropImg, x, y, size = (256, 256))
				cv2.imwrite(self.filterImgDir + 'ResizedImage' + str(counter) + '.jpg', cropImg)

				img = ut.plotLandmarks(resizeImg, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				cv2.imwrite(self.filterImgDir + 'testResizeImgLandmarks' + str(counter) + '.jpg', img)


				# pts = np.asarray(ut.packLandmarks(x, y))


				# np.savetxt(self.filterPTSDir + 'pts' + str(counter) + '.txt', pts)
				# cv2.imwrite(self.filterImgDir+ 'image' + str(counter) + '.jpg', cropImg)
				counter += 1

			 #    # imgs, landmarks = self.extract(path + "/", file)
			 #    imgs, landmarks = self.extract(self.rawDir + "/", file)   
			 #    if not self.debug:                 
			 #        self.saveImg(imgs, landmarks, file)
			 #    counter += 1

			 #    if counter % 100 == 0:
			 #        print counter
			 #        # print path

	def process(self, img, x, y):
		xMin = min(x)
		yMin = min(y)
		xMax = max(x)
		yMax = max(y)
		xMean = (xMax + xMin)/2.0
		yMean = (yMax + yMin)/2.0
		edge = max(yMax - yMin, xMax - xMin)
		# ground-truth center, W -> (center*disturbance（+-10%）, W*1.5*disturbance（+-20%）)  --》 new center, W
		newXMean = xMean * self.getDisturbance(0.1)
		newYMean = yMean * self.getDisturbance(0.1)
		newEdge = edge * 1.5 * self.getDisturbance(0.2)

		if self.debug:
			labels = [xMean, yMean, edge]
			img = ut.plotTarget(img, labels, ifSquareOnly = True)
			labels = [newXMean, newYMean, newEdge]
			img = ut.plotTarget(img, labels, ifSquareOnly = True, ifGreen = True)
			cv2.imwrite('testRectangle.jpg', img)

		cropImg = img[int(newYMean - newEdge/2.0) : int(newYMean + newEdge/2.0), int(newXMean - newEdge/2.0) : int(newXMean + newEdge/2.0)]
		x = np.asarray(x)
		y = np.asarray(y)
		x = x - int(newXMean - newEdge/2.0)
		y = y - int(newYMean - newEdge/2.0)	
		return cropImg, x, y


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
	PreProcess().run()