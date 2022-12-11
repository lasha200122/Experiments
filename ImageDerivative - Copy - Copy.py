import cv2
import numpy as np
from scipy import ndimage
import os

''''
This file is written for car speed detection. 

Input Values:
path -> video path (.mp4)
frames_per_sec -> number of frames per sec

Now Methods:
SaveFrames -> Saves Videos all frames as pictures. (inputs: path, name, form) name -> name of a picture | form -> .png ; .jpg
SaveImage -> Saves One frame as image
ReadImage -> transforms image to matrix
ReadImageAsFloat -> transforms image to float matrix (for gray images)
ShowVideo -> shows a video ;)
FrobeniusNorm -> calculates a Frobenius Norm of a matrix
NuclearNorm -> calculates a Nuclear Norm of a matrix
InfNorm -> calculates a infinity norm of a matrix
InfReverseNorm -> calculates a -infinity norm of a matrix
OneNorm -> calculates a first norm of a matrix
MinusOneNorm -> calculates a -first norm of a matrix
TwoNorm -> calculates a second norm of a matrix
MinusTwoNorm -> calculates a -second norm of a matrix
Vector2_norm -> calculates a vectors second norm
Mask -> returns masked image inputs image, mask (vector example: [-1,0,1] ) and threshold for ur choice
Magnitude -> returns masked image calculating second norm of two masked images
GetObjectFloat -> Calculates difference between two images and returns the result image
GetObject -> Finds car on image
NormDifference -> calculaates difference between two vectors 
SubstractImages -> Substracts images
zoom -> zoom in or zoom out image
DrawLines -> Draw Lines on video or image
ObjectDetection -> this is a main function it takes a video with car. then it gets the frames and mask them after that
it tries to detect the car. after car detection this method checks the starting points and ending points. we have the 
distance and we can easily calculate the speed of car. also this method calculates the FPS of the video and time duration.
ObjectDetection2 -> ObjectDetection2 does the same thing that ObjectDetection but cooler ;) 
ClearImage -> Removes noise from an image
CoolerCleaner -> also removes a noise from a picture but cooler ;) 
GetCoordinates -> gets coordinate of a point on a car
GetAverage -> calculates avarage values of a list
DrawPoint -> scatter dot on video or image
DeleteFile -> Delete File from the Folder
GetFrameCoordinates -> Gets all car pixel coordinates
PictureGrid -> Draws a grid on a picture
SplitVideo -> Splits video into two pieces


'''

class FilterObject:
    def __init__(self, path, frames_per_sec=30):
        self.path = path
        self.framesPerSec = frames_per_sec
        self.__savedPath = ""
        self.__imageName = ""
        self.__imageForm = ""

    def SaveFrames(self, path, name, form):
        video = cv2.VideoCapture(self.path)
        currentframe = 0
        while True:
            print("working.")
            print("working..")
            print("working...")
            success, frame = video.read()
            cv2.imwrite(os.path.join(path, name + str(currentframe) + form), frame)
            currentframe += 1
            if not success:
                break
        print("Done")
        self.__savedPath = path
        self.__imageName = name
        self.__imageForm = form
        return currentframe

    def SaveImage(self, img, path, name, form):
        cv2.imwrite(os.path.join(path, name + form),img)
        return

    def ReadImage(self, path):
        image = cv2.imread(path)
        return image

    def ReadImageAsFloat(self, path):
        image = cv2.imread(path, 0).astype(np.float64)
        return image

    def ShowVideo(self):
        video = cv2.VideoCapture(self.path)
        print("Press Q to close the window")
        while True:
            success, frame = video.read()
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("Done")
        return

    def FrobeniusNorm(self,matrix):
        result = np.linalg.norm(matrix,'fro')
        return result

    def NuclearNorm(self,matrix):
        result = np.linalg.norm(matrix,'nuc')
        return result

    def InfNorm(self,matrix):
        result = np.linalg.norm(matrix,'inf')
        return result

    def InfReverseNorm(self,matrix):
        result = np.linalg.norm(matrix,'-inf')
        return result

    def OneNorm(self,matrix):
        result = np.linalg.norm(matrix,'1')
        return result

    def MinusOneNorm(self,matrix):
        result = np.linalg.norm(matrix,'-1')
        return result

    def TwoNorm(self,matrix):
        result = np.linalg.norm(matrix, '2')
        return result

    def MinusTwoNorm(self,matrix):
        result = np.linalg.norm(matrix, '-2')
        return result

    def Vector2_norm(self,vector):
        return np.sqrt(sum([i*i for i in vector]))

    def Mask(self, image, mask, treshHold):
        x = treshHold * np.array(mask)
        f = ndimage.convolve(image, x)
        return f

    def Magnitude(self, image, mask1, mask2, treshhold):
        x = treshhold * np.array(mask1)
        y = treshhold * np.array(mask2)
        fx = ndimage.convolve(image,x)
        fy = ndimage.convolve(image,y)
        result = np.sqrt(fx**2 + fy**2)
        return result

    def GetObjectFloat(self, image1, image2, threshhold, color):
        #image1 is picture with object
        for i in range(len(image1)):
            if self.NormDifference(image1[i], image2[i]) < threshhold:
                image2[i] = np.array(color)
        return image2

    def GetObject(self,image1,image2,threshhold,color):
        for i in range(len(image1)):
            for j in range(len(image1[i])):
                if self.NormDifference(image1[i][j],image2[i][j]) < threshhold:
                    print(image1[i][j])
                    image2[i][j] = np.array(color)
        return image2

    def NormDifference(self, vector1, vector2):
        return abs(self.Vector2_norm(vector1) - self.Vector2_norm(vector2))

    def SubstractImages(self,img1,img2):
        result = np.array(img1) - np.array(img2)
        return result

    def zoom(self, img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

    def DrawLines(self):
        cap = cv2.VideoCapture(self.path)  # cap for capture to get a video by its name
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.line(frame, (100, 133), (600, 133), (255, 0, 0), 5)
            cv2.line(frame, (50, 233), (700, 233), (255, 0, 0), 5)
            if frame is None:
                break
            cv2.imshow("Display window", frame)
            k = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return

    def ObjectDetection(self):
        # capturing video
        capture = cv2.VideoCapture(self.path)
        img_2 = self.ReadImage("Frames/frame0.png")

        count = 0
        left = []
        right = []
        while capture.isOpened():

            # to read frame by frame
            ret, img_1 = capture.read()
            # _, img_2 = capture.read()
            if ret:
                # find difference between two frames
                diff = cv2.absdiff(img_1, img_2)

                # to convert the frame to grayscale
                # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                # diff = self.SubstractImages(img_1,img_2)
                # apply some blur to smoothen the frame
                diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

                # to get the binary image
                _, thresh_bin = cv2.threshold(diff_blur, 50, 255, cv2.THRESH_BINARY)
                wow = cv2.cvtColor(thresh_bin,cv2.COLOR_BGR2GRAY)
                img_1 = img_1[50:250,100:1500]
                wow = wow[50:250, 100:1500]
                # to find contours
                contours, hierarchy = cv2.findContours(wow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #
                cv2.line(img_1, (164, 82), (265, 82), (255, 0, 0), 5)

                cv2.line(img_1, (1355, 82), (1446, 82), (255, 0, 0), 5)
                # to draw the bounding box when the motion is detected
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) > 50:
                        cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if y == 82:
                            self.SaveImage(img_1,"C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/","Line" + str(count),".png")
                            left.append(count)
                        if y == 69:
                            self.SaveImage(img_1, "C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/",
                                           "Line" + str(count), ".png")

                            right.append(count)
                cv2.drawContours(img_1, contours, -1, (0, 255, 0), 2)
                count += 1

                cv2.imshow("Detecting Motion...", img_1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        if len(right) != 0:
            frames = abs((left[0] - right[len(right)-1]))
            time = frames / 20
            print("time: ", time)
            distance = 43  # racaa

            speed = distance / time * 3600 / 1000
            print("Speed: ", speed )
            print("frames: " , frames)

    def ObjectDetection2(self,count1,count2,count3):
        # capturing video
        capture = cv2.VideoCapture(self.path)
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        index = 0
        count1 = 0
        count2 = 0
        count3  = 0

        startBorderX = 133
        endBorderX = 600

        startFrame = None
        endFrame = None
        count = 0
        left = []
        right = []
        while capture.isOpened():
            # to read frame by frame
            sc, img_1 = capture.read()
            sc2, img_2 = capture.read()

            if sc and sc2:
                # find difference between two frames
                diff = cv2.absdiff(img_1, img_2)

                # to convert the frame to grayscale
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                # apply some blur to smoothen the frame
                diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

                # to get the binary image
                _, thresh_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

                # to find contours
                contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # to draw the bounding box when the motion is detected
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)


                    # if startFrame is None and x > startBorderX and x < endBorderX:
                    #     startFrame = index
                    #     print(startFrame)
                    # elif endFrame is None and x > endBorderX:
                    #     endFrame = index
                    #     print(endFrame)

                    if cv2.contourArea(contour) > 300:

                        cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if y == 112:
                            count1 = count
                            left.append(count1)
                        if y <= 88:
                            count2 = count
                            right.append(count2)

                # cv2.drawContours(img_1, contours, -1, (0, 255, 0), 2)
                # display the output
                count += 2
                cv2.imshow("Detecting Motion...", img_1)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
        print("Left" , left[0], "<- Right: ", right[0])
        deltaFrames = abs(left[0] - right[0])  #52 - 28 = 24
        print("Sxvaoba: ", deltaFrames)
        fps = capture.get(cv2.CAP_PROP_FPS)
        time = deltaFrames / fps
        print("time: ",time)
        distance = 40  # racaa

        speed = distance / time  * 3600 / 1000
        print("speed: ",speed, "KM/H")
        # do whatever

    def ClearImage(self, img, thr):
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] < thr:
                    img[i][j] = 0
        return img

    def CoolerCleaner(self,img1, img2, thr):
        for i in range(len(img1)):
            for j in range(len(img1[i])):
                if img1[i][j] < thr:
                    img1[i][j] = 0
                if img2[i][j] < thr:
                    img2[i][j] = 0
        return [img1,img2]

    def GetCoordinates(self,img):
        indices = np.where(img >= [50])
        coordinates = list(zip(indices[0], indices[1]))
        print(coordinates)
        #print(coordinates)
        X = [i[0] for i in coordinates]
        Y = [i[1] for i in coordinates]
        if len(X) == 0: return [0,0]
        if len(Y) == 0: return [0,0]
        print(sum(Y),len(Y))
        return [int(sum(X)/len(X)), int(sum(Y)/len(Y))]

    def GetAverage(self,ls):
        if (len(ls) == 0): return 0
        result = sum(ls)/ len(ls)
        return result

    def DrawPoint(self,img,x,y, rgb):
        img[int(x),int(y)] = rgb
        return img

    def DeleteFile(self,path):
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    #path3 empty picture
    def GetFrameCoordinates(self, path1, path2,path3):

        img1 = cv2.imread(path1,0).astype(np.float64)
        img2 = cv2.imread(path2,0).astype(np.float64)
        img3 = cv2.imread(path3,0).astype(np.float64)

        BgC_img1 = self.SubstractImages(img3, img1)
        BgC_img2 = self.SubstractImages(img3,img2)

        ls = self.CoolerCleaner(BgC_img1,BgC_img2,70)
        cleanImg1 = ls[0]
        cleanImg2 = ls[1]


        maskImg1 = self.Magnitude(cleanImg1,[[-1,2,1]],[[-1,2,1]],-1)
        maskImg2 = self.Magnitude(cleanImg2, [[-1, 2, 1]], [[-1, 2, 1]], -1)

        vacumCleaner = self.CoolerCleaner(maskImg1,maskImg2,120)
        x = vacumCleaner[0]
        y = vacumCleaner[1]
        coordinate_1 = self.GetCoordinates(x)
        coordinate_2 = self.GetCoordinates(y)

        return [coordinate_1,coordinate_2]

    def PictureGrid(self):
        capture = cv2.VideoCapture(self.path)
        while capture.isOpened():
            success, img = capture.read()

            # function to display the coordinates of
            # of the points clicked on the image
            def click_event(event, x, y, flags, params):

                # checking for left mouse clicks
                if event == cv2.EVENT_LBUTTONDOWN:
                    # displaying the coordinates
                    # on the Shell
                    print(x, ' ', y)

                    # displaying the coordinates
                    # on the image window
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(x) + ',' +
                                str(y), (x, y), font,
                                1, (255, 0, 0), 2)
                    cv2.imshow('image', img)

                # checking for right mouse clicks
                if event == cv2.EVENT_RBUTTONDOWN:
                    # displaying the coordinates
                    # on the Shell
                    print(x, ' ', y)

                    # displaying the coordinates
                    # on the image window
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    b = img[y, x, 0]
                    g = img[y, x, 1]
                    r = img[y, x, 2]
                    cv2.putText(img, str(b) + ',' +
                                str(g) + ',' + str(r),
                                (x, y), font, 1,
                                (255, 255, 0), 2)
                    cv2.imshow('image', img)

            # driver function
            if __name__ == "__main__":
                # reading the image
                # displaying the image
                cv2.imshow('image', img)

                # setting mouse handler for the image
                # and calling the click_event() function
                cv2.setMouseCallback('image', click_event)

                # wait for a key to be pressed to exit
                cv2.waitKey(0)

                # close the window
                cv2.destroyAllWindows()

    def SplitVideo(self):
        cap = cv2.VideoCapture("Video_8-2-2022/1.mp4")
        img_2 = self.ReadImage("Frames/frame0.png")
        roi12 = img_2[70:500, 150:805]
        roi22 = img_2[70:500, 825:1450]
        left = []
        right = []
        count = 0
        while (1):
            check, frame = cap.read()
            if check:
                roi = frame[70:500, 150:805]
                roi2 = frame[70:500, 825:1450]

                diff = cv2.absdiff(roi, roi12)

                # to convert the frame to grayscale
                # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                # diff = self.SubstractImages(img_1,img_2)
                # apply some blur to smoothen the frame
                diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

                # to get the binary image
                _, thresh_bin = cv2.threshold(diff_blur, 50, 255, cv2.THRESH_BINARY)
                wow = cv2.cvtColor(thresh_bin, cv2.COLOR_BGR2GRAY)

                diff2 = cv2.absdiff(roi2, roi22)

                # to convert the frame to grayscale
                # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                # diff = self.SubstractImages(img_1,img_2)
                # apply some blur to smoothen the frame
                diff_blur2 = cv2.GaussianBlur(diff2, (5, 5), 0)

                # to get the binary image
                _, thresh_bin2 = cv2.threshold(diff_blur2, 50, 255, cv2.THRESH_BINARY)
                wow2 = cv2.cvtColor(thresh_bin2, cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(wow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours2, hierarchy2 = cv2.findContours(wow2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) > 50:
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # if y == 82:
                            # self.SaveImage(img_1, "C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/",
                            #                "Line" + str(count), ".png")
                        left.append(count)
                        # if y == 69:
                        #     # self.SaveImage(img_1, "C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/",
                        #     #                "Line" + str(count), ".png")
                        #
                        #     right.append(count)
                cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)

                for contour in contours2:
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) > 50:
                        cv2.rectangle(roi2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # if y == 82:
                        #     self.SaveImage(img_1, "C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/",
                        #                    "Line" + str(count), ".png")
                        #     left.append(count)
                        # if y == 69:
                            # self.SaveImage(img_1, "C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/Line/",
                            #                "Line" + str(count), ".png")

                        right.append(count)
                cv2.drawContours(roi2, contours2, -1, (0, 255, 0), 2)
                count += 1


                cv2.imshow('result', roi)
                cv2.imshow('result2', roi2)

                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
            else:
                break
        if len(right) != 0:
            frames = abs((left[0] - right[len(right)-1]))
            time = frames / 20
            print("time: ", time)
            distance = 43  # racaa

            speed = distance / time * 3600 / 1000
            print("Speed: ", speed )
            print("frames: " , frames)



# model = FilterObject("Video_8-2-2022/1.mp4")
# model.SaveFrames("GifFrames/","Fram1",".png")
# model.SplitVideo()
# model.ObjectDetection()
# model.ShowVideo()
# model.SaveFrames("Frames/","frame", ".png")
