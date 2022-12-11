import cv2
import numpy as np
from scipy import ndimage
from skimage import color

path ="Gifs/1.gif"
class Test:
    def __init__(self,path):
        self.path = path
        self.frames = []
        self.Video()
        points = self.AxisPoints(self.frames)  # [(x_min, x_max), (y_min, y_max)] output
        self.VideoWithAxis(points)


    def FirstDerivative(self, image, mask, treshHold):
        x = treshHold * np.array(mask)
        f = ndimage.convolve(image, x)
        return f

    def Video(self):
        video = cv2.VideoCapture(self.path,0)
        print("Press Q to close the window")
        while True:
            success, frame = video.read()
            if success:
                # img = frame[170:255,48:430]
                first = self.FirstDerivative(color.rgb2gray(frame),[[-1,0,1],[-1,0,1],[-1,0,1]],-1)
                second = self.FirstDerivative(color.rgb2gray(frame), [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], -1)
                gray = np.sqrt(first**2 + second**2)

                self.frames.append(gray)
                cv2.imshow("camera", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        print("Done")
        return

    def AxisPoints(self,frames):
        coordinates = []
        for frame in frames:
            for y in range(len(frame)):
                for x in range(len(frame[y])):
                    if frame[y][x] >= .5:
                        coordinates.append((y,x))
        x_min = min(coordinates, key = lambda  t: t[1])[1]
        x_max = max(coordinates, key = lambda  t: t[1])[1]
        y_min = min(coordinates, key=lambda t: t[0])[0]
        y_max = max(coordinates, key=lambda t: t[0])[0]
        return [(x_min,x_max),(y_min,y_max)]

    def VideoWithAxis(self,points):
        video = cv2.VideoCapture(self.path, 0)
        print("Press Q to close the window")
        while True:
            success, frame = video.read()
            if success:
                first = self.FirstDerivative(color.rgb2gray(frame), [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], -1)
                second = self.FirstDerivative(color.rgb2gray(frame), [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], -1)
                gray = np.sqrt(first ** 2 + second ** 2)
                print(gray.shape)
                cv2.line(gray, (points[0][0], points[1][0]), (points[0][0], points[1][1]), [255, 255, 255], 2)
                cv2.line(gray, (points[0][0], (points[1][1] - points[1][0])//2 + points[1][0]), (points[0][1],(points[1][1] - points[1][0])//2 + points[1][0]), [255, 255, 255], 2)
                cv2.imshow("camera2", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        print("Done")
        return

Test(path)