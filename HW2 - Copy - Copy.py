import random
import matplotlib.pyplot as plt
import numpy as np

arr = []
# n = int(input("input number of students:"))

for j in range(100):
    stud_mat = []
    for i in range(4):
        row = [random.randint(51, 100) for x in range(5)]
        # print("row:", row)
        stud_mat.append(row)
        # print("mat:",stud_mat)
    arr.append(stud_mat)

print(arr)
# for i in range( len(arr)):
#     print(arr[i])


normsfro = []
for i in range(len(arr)):
    normsfro.append(int(np.linalg.norm(arr[i], "fro")))
# forbenious norm is the best because it uses all the data

print()
print(normsfro)
print(sorted(normsfro))

# srtd= sorted(arr,key=lambda x: np.linalg.norm(x, "fro"))
# print(srtd)

# norms2=[]
# for i in range(len(arr)):
#     norms2.append(int(np.linalg.norm(arr[i], 2)))
#
# print()
# print(norms2)


plt.figure(figsize=(14, 10))
plt.scatter(normsfro, normsfro, marker="+", c='red')
plt.show()
