import copy
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array


def mod(x):
    if x < 0:
        x = -x
    return x


def swap(list1, list2):
    save = 0

    oldsub = mod(sum(list1) - sum(list2))

    for i in range(0, len(list1)):
        for j in range(0, len(list2)):
            oldsub = mod(sum(list1) - sum(list2))
            save = list1[i]
            list1[i] = list2[j]
            list2[j] = save

            newsub = mod(sum(list1) - sum(list2))

            if newsub >= oldsub:
                list2[j] = list1[i]
                list1[i] = save

    return [list1, list2]


def helper(list1, list2):
    while list2 != list1:
        list2 = list1

        for i in range(0, len(list1)):
            for j in range(0, len(list1[0])):
                swap(list1[i], list1[j])

    return list1


def freshemen(list1):
    list2 = []

    for i in range(0, len(list1)):
        for j in range(0, len(list1[0])):
            k = sum(list1[i][j])
            list1[i][j] = k
    for i in range(0, len(list1)):
        list2.append(copy.deepcopy(list1[i]))
    print(list1, "list1")
    list2[0][0] = -2

    print(list1, "list1")
    print(list2, "list2")

    answer = helper(list1, list2)

    return answer

def gpa(list):
    colavarage = 0
    rowavarage = 0
    newlist = []
    for i in range(0, len(list)):
        for j in range(0, 4):
            for k in range(0, 5):
                colavarage = colavarage + list[i][j][k]



            colavarage = colavarage / 5

            rowavarage = rowavarage + colavarage
            colavarage=0


        newlist.append(rowavarage / 4)
        rowavarage = 0
    newlist.sort()
    return newlist[len(newlist)-1]

def norm(x):
    return sum(x) / 3


def compare(i, x, y, z):
    return min(mod(i - x), mod(i - y), mod(i - z))


def clustering(listofplayers):
    answer = []

    centroid1 = 80
    centroid2 = 90
    centroid3 = 75

    first_league = []
    second_league = []
    third_league = []

    for i in range(0, len(listofplayers)):
        if compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(norm(listofplayers[i]) - centroid1):
            first_league.append(listofplayers[i])
        elif compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(
                norm(listofplayers[i]) - centroid2):
            second_league.append(listofplayers[i])
        elif compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(
                norm(listofplayers[i]) - centroid3):
            third_league.append(listofplayers[i])

    answer.append(first_league)
    answer.append(second_league)
    answer.append(third_league)
    samelist = copy.deepcopy(listofplayers)
    samelist[0][0] = -1
    print(samelist)
    print(listofplayers)

    while samelist != answer:
        samelist = answer
        centroid1 = sum(first_league[0])/ len(first_league)
        centroid2 = sum(second_league[0]) / len(second_league)
        centroid3 = sum(third_league[0]) / len(third_league)
        for i in range(0, len(listofplayers)):
            if compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(
                    norm(listofplayers[i]) - centroid1):
                first_league.append(listofplayers[i])
            elif compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(
                    norm(listofplayers[i]) - centroid2):
                second_league.append(listofplayers[i])
            elif compare(norm(listofplayers[i]), centroid1, centroid2, centroid3) == mod(
                    norm(listofplayers[i]) - centroid3):
                third_league.append(listofplayers[i])

    answer.append(first_league)
    answer.append(second_league)
    answer.append(third_league)

    return answer


lst1 = array("download.jpg")
lst2 = array("images.jpg")





def  helper(img1,img2,size):
    new1 = Image.open(img1)
    new1 = new1.resize(size)
    new1.save(img1)
    new2 = Image.open(img2)
    new2 = new2.resize(size)
    new2.save(img2)
def blend(l1, l2, r):

    make_two_phots_same_size(l1,l2,(200,300))

    answer = imageio.v2.imread("priject.2.png")


    l2 = imageio.v2.imread(l2)
    l1 = imageio.v2.imread(l1)

    for t in range(0, 10):


          r = r + 0.1
          plt.imshow(answer)
          plt.savefig("priject.2.png")

    for i in range(0, len(l1)):


        for j in range(0, len(l1[0])):

            k = 0
            while k < 3:

                f = int(float(l1[i][j][k]) * (1 - r))
                s = int(float(l2[i][j][k]) * r)
                answer[i][j][k] = int(f + s)

                k += 1
                print(k)
                print(answer[i][j][k])

    return answer
