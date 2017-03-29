import cv2
import numpy as np
import utility as ut


imgName = "testFiles/image_100_05.jpg"
img = cv2.imread(imgName)
print img.shape
(w, h, _) = img.shape

# FTr = open("testFiles/testImg.txt",'r')
# DataTr = FTr.readlines()
# TrNum = len(DataTr)



# strLine = DataTr[0]
# strCells = strLine.rstrip(' \n').split(' ')
# # print "strCells: ", strCells
# # print "len(strCells): ", len(strCells)
# # raise "debug"
# imgName = strCells[0]


# labels = np.array(strCells[1:]).astype(np.float)
# labelsPTS=labels[:136].reshape([68,2])


# x, y = ut.unpackLandmarks(labelsPTS, 256)

# img = ut.plotLandmarks(img, x, y, 256, name = None, ifRescale = False, ifReturn = True)

# # print imgName
# # print labelsPTS.shape

# resizeImg, x, y = ut.resize(img, x, y, random = True)

# print type(resizeImg)
# print resizeImg.shape
# resizeImg = ut.plotLandmarks(resizeImg, x, y, 256, name = None, ifRescale = False, ifReturn = True)

# cv2.imshow("resize", resizeImg)

# cv2.imshow("original",img)

# cv2.waitKey(0)




import PIL
from PIL import Image

basewidth = 1000
# img = Image.open('somepic.jpg')

# wpercent = (basewidth/float(img.size[0]))
wpercent = basewidth/float(w)

print wpercent

hsize = int((h*float(wpercent)))

print basewidth,hsize

img = cv2.resize(img,(basewidth,hsize))

# img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
cv2.imwrite('sompic.jpg', img) 















