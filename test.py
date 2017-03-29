# # # def deNormalize(array):
# # #     if isinstance(array, list):
# # #         array = list(array)
# # #         newArray = []
# # #         for i in range(len(array)):
# # #             newArray.append((array[i] + 0.5) * 128.0)
# # #         return newArray
# # #     else:
# # #         return (array+ 0.5) * 128.0


# # # def normalize(array):
# # #     if isinstance(array, list):
# # #         array = list(array)
# # #         newArray = []
# # #         for i in range(len(array)):
# # #             newArray.append((array[i]/128.0) - 0.5)
# # #         return newArray
# # #     else:
# # #         return (array/128.0) - 0.5

# # # a = [1, 2, 3]

# # # a = normalize(a)
# # # print a 
# # # print deNormalize(a)



# # # a = 30


# # # b = 10


# # # print normalize(a - b)
# # # print normalize(a) - normalize(b)


# # batch_size, imSize


# # def DataGenBB(DataStrs, train_start,train_end):

# #     generateFunc = ["rotate", "resize"]

# #     InputData = np.zeros([batch_size, imSize, imSize, 3], dtype = np.float32)
# #     InputLabel = np.zeros([batch_size, 7], dtype = np.float32)

# import random
# tag = random.choice(['ass', 'fadf'])

# print tag
# ---------------------------------------------------------------------------------------

import cv2
import numpy as np
import utility as ut

# imgName = "testFiles/995.jpg"
imgName = "testFiles/image_100_05.png"
img = cv2.imread(imgName)
print img.shape
(w, h, _) = img.shape

# FTr = open("testFiles/995.txt",'r')
FTr = open("testFiles/testImg.txt",'r')
DataTr = FTr.readlines()
print DataTr

TrNum = len(DataTr)

strLine = DataTr[0]
strCells = strLine.rstrip(' \n').split(' ')
# print "strCells: ", strCells
# print "len(strCells): ", len(strCells)
# raise "debug"
imgName = strCells[0]


labels = np.array(strCells[1:]).astype(np.float)
labelsPTS=labels[:136].reshape([68,2])


x, y = ut.unpackLandmarks(labelsPTS, 256)

# img = ut.plotLandmarks(img, x, y, 256, name = None, ifRescale = False, ifReturn = True)

# cv2.imshow("resize", img)
# cv2.waitKey(0)
# cv2.imwrite("original.jpg",img)
# raise "debug"

# print imgName
# print labelsPTS.shape
counter = 0
while True:

	# imgName = "testFiles/image_100_05.png"
	# img = cv2.imread(imgName)
	# print img.shape
	# (w, h, _) = img.shape
	# resizeImg = None
	x, y = ut.unpackLandmarks(labelsPTS, 256)

	resizeImg, x, y = ut.resize(img, x, y, random = True)

	print type(resizeImg)
	print resizeImg.shape
	
	resizeImg = ut.plotLandmarks(resizeImg, x, y, name = None, ifRescale = False, ifReturn = True)

	# cv2.imshow("resize"+ str(counter), resizeImg)

	# cv2.imshow("original",img)

	# cv2.waitKey(0)

	cv2.imwrite("resizeExperiment" + str(counter) + ".jpg",resizeImg)

	counter += 1
	if counter > 5:
		break

# import PIL
# from PIL import Image

# basewidth = 300
# # img = Image.open('somepic.jpg')
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
# img.save('sompic.jpg') 

