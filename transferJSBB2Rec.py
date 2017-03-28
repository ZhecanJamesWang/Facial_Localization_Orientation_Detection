import os
import cv2
import numpy as np
debug = False

def plotTarget(image, labels, ifGreen = False):
    img = np.copy(image)
    if ifGreen:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 255, 0), 3)
    else:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 0, 255), 3)
    return img


# rawDir = "competitionImageDataset/testset/profile/"
rawDir = "competitionImageDataset/testset/semifrontal/"

files = os.listdir(rawDir)
print len(files)
counter = 0
for fileName in files:
	if ".JSBB_Select" in fileName:
		fileHeader = fileName.split(".")[0]		
		fr = open(rawDir + fileName, 'r')
		index = int(fr.readlines()[0])
		print "index: ", index
		if index > 0:
			index = index - 1

		# if not os.path.isfile(rawDir + fileHeader + '.rec'):
		if True:
		# if index < 0:
			print "fileName: ", fileName
			print "index: ", index
			print "fileHeader: ", fileHeader
			if index < 0:
				print "JSBBFile = fileHeader + .JSBB_Update"
				JSBBFile = fileHeader + ".JSBB_Update"
			else:
				print "JSBBFile = fileHeader + .JSBB"
				JSBBFile = fileHeader + ".JSBB"
			fr = open(rawDir + JSBBFile, 'r')
			lines = fr.readlines()
			if index < 0:
				boudingBoxInfo = lines[0]
			else:
				boudingBoxInfo = lines[index]
			print "len(lines): ", len(lines)
			print "lines: ", lines
			print "boudingBoxInfo ", boudingBoxInfo
			print "boudingBoxInfo ", boudingBoxInfo.split("\t")
			print "boudingBoxInfo ", type(boudingBoxInfo)

			if index < 0:			
				confidence, xMin, yMin, xMax, yMax = boudingBoxInfo.split("\t")
				confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax[:-1])
			else:
				try:
					confidence, xMin, yMin, xMax, yMax = boudingBoxInfo.split(" ")
					confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax)
				except Exception as e:
					confidence, xMin, yMin, xMax, yMax = boudingBoxInfo.split("\t")
					confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax[:-1])
	
			print "xMin, yMin, xMax, yMax: ", xMin, yMin, xMax, yMax

			if debug:
				imgName = fileHeader + ".jpg"
				img = cv2.imread(rawDir + imgName)
				plotImg = plotTarget(img, [xMin, yMin, xMax, yMax])
				cv2.imshow("img", plotImg)
				cv2.waitKey(0)
				raise "debug"

			# if xMax - xMin != yMax - yMin:

			xMean = (xMax + xMin)/2.0
			yMean = (yMax + yMin)/2.0
			edge = max(xMax - xMin, yMax - yMin)
			xMin = xMean - edge/2.0
			xMax = xMean + edge/2.0
			yMin = yMean - edge/2.0
			yMax = yMean + edge/2.0

			fw = open(rawDir + fileHeader + '.rec','w')
			print rawDir + fileHeader + '.rec'
			fw.write(str(xMin) + " " + str(yMin) + " " + str(xMax) + " " + str(yMax))
			print "writing into:", str(xMin) + " " + str(yMin) + " " + str(xMax) + " " + str(yMax)
			counter += 1
			# if counter % 100 == 0:
			print "counter: ", counter


