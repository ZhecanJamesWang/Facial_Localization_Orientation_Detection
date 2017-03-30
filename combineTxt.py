

f2 = open("/home/james/Facial_Localization_Orientation_Detection/dataKBKC4_train.txt", 'a')
lines = f2.readlines()
print "f2: ", len(lines)

f1 = open("/home/james/Facial_Localization_Orientation_Detection/data/Menpo39TrainProcessed/menpo39Data.txt", 'r')
f2 = open("/home/james/Facial_Localization_Orientation_Detection/dataKBKC4_train.txt", 'a')
lines = f1.readlines()
for line in lines:
	print line
	f2.write(line)
f2.close()
f1.close()

f2 = open("/home/james/Facial_Localization_Orientation_Detection/dataKBKC4_train.txt", 'a')
lines = f2.readlines()
print "f2: ", len(lines)