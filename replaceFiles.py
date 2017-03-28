import glob

files = glob.glob("competitionImageDataset/testset/semifrontal/*.JSBBM")
for file in files:
	print "file: ", file
	fileHeader = file.split(".")[0]
	fr = open(file, 'r')
	fw = open(fileHeader + '.JSBB','w') 
	lines = fr.readlines()
	for line in lines:
		line = line.split(" ")
		content = ""
		for index in range(len(line) - 1):
			element = line[index]
			content += (element + " ")
		content += line[-1] 
		print "content: ", content
		fw.write(content)
