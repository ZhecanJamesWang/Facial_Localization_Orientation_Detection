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


# rawDir = "data/competitionImageDataset/testset/profile/"
rawDir = "data/competitionImageDataset/testset/semifrontal/"
outputDir = "data/preProcessedSemifrontal/"
# outputDir = "data/preProcessedProfile/"


files = os.listdir(rawDir)
print len(files)
counter = 0
for fileName in files:
	if ".rec" in fileName:
		fileHeader = fileName.split(".")[0]		
		fr = open(rawDir + fileName, 'r')
		lines = fr.readlines()
		imgName = fileHeader + ".jpg"
		img = cv2.imread(rawDir + imgName)
		w, h, _ = img.shape
		if debug:
			cv2.imshow("original", img)
		xMin, yMin, xMax, yMax = lines[0].split(" ")
		xMin, yMin, xMax, yMax = float(xMin), float(yMin), float(xMax), float(yMax)
		
		if debug:
			plotImg = plotTarget(img, [xMin, yMin, xMax, yMax])
			cv2.imshow("img", plotImg)

		xMean = (xMax + xMin)/2.0
		yMean = (yMax + yMin)/2.0
		edge = max(xMax - xMin, yMax - yMin)
		newEdge = 1.3 * edge
		newXMin = xMean - newEdge/2.0
		newXMax = xMean + newEdge/2.0
		newYMin = yMean - newEdge/2.0
		newYMax = yMean + newEdge/2.0

		# xMin = xMin - edge/2.0
		# yMin = yMin - edge/2.0
		# xMax = xMax - edge/2.0
		# yMax = yMax - edge/2.0
		# edge = max(xMax - xMin, yMax - yMin)

		if newXMin < 0:
			newXMin = 0
		if newYMin < 0:
			newYMin = 0
		if newXMax > w:
			newXMax = w
		if newYMax > h:
			newYMax = h

		cropImg = img[newYMin : newYMax, newXMin:newXMax]
		label = np.asarray([xMean, yMean, edge])
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

		if debug:
			cv2.imshow("cropped", cropImg)
			cv2.waitKey(0) 

		np.savetxt(outputDir + fileHeader + '.txt', label)
		cv2.imwrite(outputDir + fileHeader + '.jpg', cropImg)
		counter += 1

		if counter % 100 == 0:
			print "counter: ", counter

		# print "index: ", index
		# if index > 0:
			# index = index - 12