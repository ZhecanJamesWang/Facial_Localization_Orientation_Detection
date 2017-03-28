import os

rawDir = "competitionImageDataset/testset/semifrontal/"
files = os.listdir(rawDir)
print len(files)
missCounter = 0
existCounter = 0
for fileName in files:
	if ".jpg" in fileName:
		fileHeader = fileName.split(".")[0]
		recName = fileHeader + ".JSBB_Select"
		if os.path.isfile(rawDir + recName):
			existCounter += 1
		else:
			missCounter += 1
		# f = open(rawDir + fileName, 'r')
print "missCounter: ", missCounter
print "existCounter: ", existCounter

		# print "deleteCounter: ", deleteCounter
		# lines = f.readlines()
		# if counter%10 == 0:
		# 	print lines
		# if int(lines[0]) == -1:
		# 	os.remove(rawDir + fileName)
		# 	deleteCounter += 1
		# counter += 1