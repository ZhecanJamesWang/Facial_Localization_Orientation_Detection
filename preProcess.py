# -*- coding: utf-8 -*-
from keras import optimizers
import utility as ut
import numpy as np
import selfModel as m
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreProcess(object):
	def __init__(self):
		self.rawDataDir = "/home/james/Menpo39_Valid"

	def getDataByFiles(self):
	    counter = 0
	    files = os.listdir(self.rawDataDir)

	    for file in files:
	        if file != ".DS_Store" and self.format in file:
	        	if ".pts39" in file:
	        		print "file name: ", file
	        	# P39=np.loadtxt(pts39)
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


		
if __name__ == '__main__':
	PreProcess().run()