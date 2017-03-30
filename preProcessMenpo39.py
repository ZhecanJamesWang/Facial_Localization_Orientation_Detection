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
		self.rawDataDir = "data/Menpo39_Train/"
		self.outputDir = "data/Menpo39TrainProcessed/"	
		self.init = True
		self.debug = False
		self.imSize = 256

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
				
				if self.debug:
					cv2.imshow("originalImg", img)


				img, x, y = ut.resize(img, x, y, random = False, size = (self.imSize, self.imSize))
				# img, x, y = ut.scale(img, x, y, self.imSize)

				print "after resize img.shape: ", img.shape
				# if self.debug:
				# img = ut.plotLandmarks(img, x, y, imSize = self.imSize, ifReturn = True, circleSize = 2)

				if self.debug:
					cv2.imshow("resizedImg", img)
					cv2.waitKey(0)

				cv2.imwrite(self.outputDir + fileHeader + '.jpg', img)
				
				info = ""
				info += (self.outputDir + fileHeader + '.jpg')
				for index in range(len(x)):
					info += (" " + str(x[index]))
					info += (" " + str(y[index]))

				if os.path.exists(self.outputDir + 'menpo39Data.txt') and self.init == False:
					f = open(self.outputDir + 'menpo39Data.txt', 'a')
				else:
					f = open(self.outputDir + 'menpo39Data.txt','w')
					self.init = False

				f.write(info)
				f.write('\n')
				f.close()


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