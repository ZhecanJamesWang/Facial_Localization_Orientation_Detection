# # def deNormalize(array):
# #     if isinstance(array, list):
# #         array = list(array)
# #         newArray = []
# #         for i in range(len(array)):
# #             newArray.append((array[i] + 0.5) * 128.0)
# #         return newArray
# #     else:
# #         return (array+ 0.5) * 128.0


# # def normalize(array):
# #     if isinstance(array, list):
# #         array = list(array)
# #         newArray = []
# #         for i in range(len(array)):
# #             newArray.append((array[i]/128.0) - 0.5)
# #         return newArray
# #     else:
# #         return (array/128.0) - 0.5

# # a = [1, 2, 3]

# # a = normalize(a)
# # print a 
# # print deNormalize(a)



# # a = 30


# # b = 10


# # print normalize(a - b)
# # print normalize(a) - normalize(b)


# batch_size, imSize


# def DataGenBB(DataStrs, train_start,train_end):

#     generateFunc = ["rotate", "resize"]

#     InputData = np.zeros([batch_size, imSize, imSize, 3], dtype = np.float32)
#     InputLabel = np.zeros([batch_size, 7], dtype = np.float32)

import random
tag = random.choice(['ass', 'fadf'])

print tag
