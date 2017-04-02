from random import shuffle




f1 = open("/home/james/Facial_Localization_Orientation_Detection/data/Menpo39TrainProcessed/menpo39Data.txt", 'r')
lines1 = f1.readlines()
print "f1: ", len(lines1)


f2 = open("/home/james/Facial_Localization_Orientation_Detection/data/KBKC4_train.txt", 'r')
lines2 = f2.readlines()
print "f2: ", len(lines2)

shuffle(lines1)
shuffle(lines2)


lines2 = lines2[:int(len(lines2)* 1)]
print "after cut f2: ", len(lines2)

f3 = open("/home/james/Facial_Localization_Orientation_Detection/data/combineDataFull.txt", 'w')

for line in lines1:
	f3.write(line)

for line in lines2:
	f3.write(line)

f1.close()
f2.close()
f3.close()

f3 = open("/home/james/Facial_Localization_Orientation_Detection/data/combineDataFull.txt", 'r')
lines3 = f3.readlines()
print "f3: ", len(lines3)
f3.close()
