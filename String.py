import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color
import math
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython import display
from scipy.interpolate import CubicSpline

#ezgif.com-gif-maker.gif
path = "ezgif.com-gif-maker.gif"

class String:
    def __init__(self, path):
        # Gif/Video Path
        self.path = path
        # save Gif frames to modify
        self.frames = []
        # All data point style -> [(x,y),...]
        self.coordinates = {}
        self.key = 0
        self.filteredCoordinates = []
        self.amplitudes = []
        # runs Gif/Video modifies frames and then appends it to self.frames
        self.Video()
        #Get points for axis format -> [(x_min, x_max), (y_min, y_max)]
        points = self.AxisPoints()
        #Draw lines on video and saves all data points
        self.VideoWithAxis(points)
        for i in self.coordinates.keys():
            self.Scatter(i)
            A = self.Interpolation()
            self.amplitudes.append(A)
            self.filteredCoordinates = []
        print(self.amplitudes)

        #Create Gif
        # self.GenerateGif(-2,2,2,0,2*np.pi)

    def FirstDerivative(self, image, mask, treshHold):
        x = treshHold * np.array(mask)
        f = ndimage.convolve(image, x)
        return f

    def Video(self):
        video = cv2.VideoCapture(self.path,0) # there 0 is for Gifs
        print("Press Q to close the window")
        count = 0
        while True:
            success, frame = video.read()
            if success:
                # img = frame[170:255,48:430] -> this method can be used for cropping image
                first = self.FirstDerivative(color.rgb2gray(frame.astype(np.float64)),[[-1, 0, 1]],-1)  #first mask
                # first = 255 - color.rgb2gray(frame.astype(np.float64))
                count += 1
                # second = self.FirstDerivative(color.rgb2gray(frame.astype(np.float64)), [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 1) # second mask
                # gray = np.sqrt(first**2 + second**2) # Combination
                cv2.imwrite("C:/Users/KiuAdmin/PycharmProjects/NumericalAnalysis/GifFrames/frame" + str(count) + ".png",
                            first)
                self.frames.append(first)
                cv2.imshow("camera", first)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        print("Done")
        cv2.destroyAllWindows()
        return

    def AxisPoints(self):
        # list for storing data
        coordinates = []
        count = 0
        #for every frame
        for frame in self.frames:
            self.coordinates[count] = []
            self.key = count
            # because every frame is 2D matrix : example -> [[0,0,....],....]
            for y in range(len(frame)):
                for x in range(len(frame[y])):
                    if frame[y][x] >= 125:
                        self.coordinates[count].append((x, y))
                    # if value (from 0 to 1) is greater than 0.5 save data
                    if frame[y][x] > 0.0:
                        coordinates.append((y,x))
            count+= 1

        x_min = min(coordinates, key = lambda t: t[1])[1] # minimum of x-axis
        x_max = max(coordinates, key = lambda t: t[1])[1] # maximum of x-axis
        y_min = min(coordinates, key=lambda t: t[0])[0] # minimum of y-axis
        y_max = max(coordinates, key=lambda t: t[0])[0] # maximum of y-axis
        return [(x_min,x_max),(y_min,y_max)]

    def VideoWithAxis(self,points):
        video = cv2.VideoCapture(self.path, 0)
        print("Press Q to close the window")
        while True:
            success, frame = video.read()
            if success:
                first = self.FirstDerivative(color.rgb2gray(frame.astype(np.float64)),[[-1, 0, 1]],-1)
                # second = self.FirstDerivative(color.rgb2gray(frame), [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], -1)
                # gray = np.sqrt(first ** 2 + second ** 2)
                #Draw lines
                cv2.line(first, (points[0][0], points[1][0]), (points[0][0], points[1][1]), [255, 255, 255], 2)
                cv2.line(first, (points[0][0], (points[1][1] - points[1][0])//2 + points[1][0]), (points[0][1],(points[1][1] - points[1][0])//2 + points[1][0]), [255, 255, 255], 2)
                cv2.imshow("camera2", first)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        print("Done")
        return

    def sort(self,x,y):
        for i in range(len(x) -1):
            for j in range(i,len(x)):
                if x[i] > x[j]:
                    x[i], x[j] = x[j], x[i]
                    y[i], y[j] = y[j], y[i]
        nx = []
        ny = []
        for i in range(len(x)-1):
            if x[i] == x[i+1]:
                continue
            else:
                nx.append(x[i])
                ny.append(y[i])


        return (nx,ny)

    def Interpolation(self):
        result = self.sort(self.filteredCoordinates[0][0],self.filteredCoordinates[0][1])
        x = np.array(result[0])
        y = np.array(result[1])

        if (len(x) <5):
            return (0,0)

        cs = CubicSpline(x,y,bc_type='natural')

        nx = np.arange(x[0], x[len(x)-1],0.1)
        ny = cs(nx)
        plt.scatter(x,y,c='red')
        plt.plot(nx,ny)
        plt.show()

        T = abs(x[len(x)-1] - x[0])
        A = ny[0]

        return (A,T)

    def Scatter(self,frame):
        x = []
        y = []
        coordintes = self.coordinates[frame]
        while len(coordintes) != 0:
            count = 0
            sum = 0
            item = coordintes[0]
            for i in coordintes:
                if i[1] == item[1]:
                    count += 1
                    sum += i[0]
                    coordintes.remove(i)
            x.append(sum/count)
            y.append(item[1])
        self.filteredCoordinates.append((x, y))
        # plt.scatter(x,y)
        # plt.show()
        return

    def Plot(self):
        frame = max(self.coordinates, key=lambda k: len(self.coordinates[k]))
        x = [i[0] for i in self.coordinates[frame]]
        y = [i[1] for i in self.coordinates[frame]]
        print(frame)
        plt.plot(x, y)
        return plt.show()

    def PlotFrames(self):
        for s in range(self.key+1):
            if (len(self.coordinates[s]) == 0):
                del self.coordinates[s]
                self.key-=1
        i, j = 0, 0
        print(self.coordinates)
        PLOTS_PER_ROW = 5
        fig, axs = plt.subplots(math.ceil(self.key+1/ PLOTS_PER_ROW), PLOTS_PER_ROW, figsize=(20, 60))
        for col in list(self.coordinates.keys()):
            x = [z[0] for z in self.coordinates[col]]
            y = [z[1] for z in self.coordinates[col]]
            axs[i][j].scatter(x, y, s=1)
            axs[i][j].set_ylabel(col)
            j += 1
            if j % PLOTS_PER_ROW == 0:
                i += 1
                j = 0

        print(len(self.coordinates))
        return plt.show()

    def GenerateGif(self,amplitude_start,amplitude_end,T,a,b):
        domain = np.arange(a,b+0.1,0.1)
        amplitudes = np.arange(amplitude_start,amplitude_end,0.1)
        fig = plt.figure()
        l, = plt.plot([],[])

        plt.xlim(a,b)
        plt.ylim(-5,5)
        plt.xticks([])
        plt.yticks([])

        def func(x,a,t):
            return a*np.cos(np.pi*2*x/t)


        metadata = dict(title="String", artist="Group")
        writer = PillowWriter(fps=15,metadata=metadata)
        print(domain)
        with writer.saving(fig, "string.gif",100):
            for amplitude in amplitudes:
                y = func(domain,amplitude,T)
                l.set_data(domain,y)
                writer.grab_frame()
            for amplitude in reversed(amplitudes):
                y = func(domain, amplitude, T)
                l.set_data(domain, y)

                writer.grab_frame()
String(path)