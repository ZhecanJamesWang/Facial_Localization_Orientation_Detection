# -*- coding: utf-8 -*-
import utility as ut
import numpy as np
# import selfModel as m
import cv2
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreProcess(object):
	def __init__(self):
		self.rawDataDir = "/home/james/Menpo39_Valid/"

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
				img = ut.plotLandmarks(img, x, y, ifRescale = False, ifReturn = True, circleSize = 3)
				cv2.imwrite('test.jpg', img)
				raise "debug"
				

				
			 #    # imgs, landmarks = self.extract(path + "/", file)
			 #    imgs, landmarks = self.extract(self.rawDir + "/", file)   
			 #    if not self.debug:                 
			 #        self.saveImg(imgs, landmarks, file)
			 #    counter += 1

			 #    if counter % 100 == 0:
			 #        print counter
			 #        # print path



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