import os

rawDir = "competitionImageDataset/testset/semifrontal/"
files = os.listdir(rawDir)
print len(files)
counter = 0
deleteCounter = 0
for fileName in files:
	if ".JSBB_Select" in fileName:
		fileHeader = fileName.split(".")[0]
		# print fileName
		# print fileHeader
		# raise "debug"
		# imgName = fileHeader + ".jpg"
		# img = cv2.imread(self.rawDir + imgName)
		# w, h, _ = img.shape

		f = open(rawDir + fileName, 'r')
		lines = f.readlines()
		# if counter%10 == 0:
		# 	print lines
		# print int(lines[0])
		# raise "debug"
		if int(lines[0]) == -1:
			print os.path.isfile(rawDir + fileHeader + ".JSBB_Update")
			os.remove(rawDir + fileHeader + ".JSBB_Update")
			deleteCounter += 1
		counter += 1

print "counter: ", counter
print "deleteCounter: ", deleteCounter