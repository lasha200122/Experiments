import cv2
import numpy as np
from scipy.ndimage import filters
from scipy import optimize

class WaveEquation:
    def __init__(self, path, L):
        self.time = 0
        self.dt = 0
        self.path = path
        self.shape = (0,0)
        self.start = (0,0)
        self.end = (0,0)
        self.frames = {}
        self.grid = []
        self.constants = []
        self.GetData()
        self.Points()
        #### Printing Part ###
        print("Staring Point: " + str(self.start) + " Ending points: " + str(self.end))
        print("Duration Of Video: " + str(self.time))
        print("Delta t: " + str(self.dt))
        print("Length: " + str(L))
        self.diff = abs(self.start[1] - self.end[1])
        self.coordinateLength = L / self.diff
        self.GetGrid()
        self.dx = self.diff * self.coordinateLength / len(self.grid[0])
        self.Constant()
        self.C2 = np.sqrt(sum(self.constants) / len(self.constants))
        self.speed = np.sqrt(self.C2 * np.power(self.dx,2) / np.power(self.dt,2))

        print("Difference between coordinates: " + str(self.diff))
        print("One coordinate Length: " + str(self.coordinateLength))
        print("Delta x: " + str(self.dx))
        print("Constant: " + str(self.C2))
        print("Speed: " + str(self.speed))
        pass

    def GetData(self):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = frame_count / fps
        self.dt = self.time / frame_count
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        print("Getting Data")
        print("Frame count: " + str(frame_count))
        count = 0
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret ==True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Laplacian(frame, -1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                if count < 1:
                    print("Shape of frame: " + str(edges.shape))
                    print("Reading frames...")
                    self.shape = edges.shape
                count += 1
                edges[edges != 255] = 0
                # edges[243,80:575] = 255
                self.frames[count] = edges
                cv2.imwrite("Picture.png",edges)
                cv2.imshow("test" + str(count) + ".png",edges)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        print()
        print("Finished reading frames")
        return


    def Points(self):
        print()
        print("Finding starting and ending points coordinates")
        a = None
        b = None
        index = 0
        while a == None or b == None:
            if a == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][index] == 255:
                        a = (i,index)
            if b == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][len(self.frames[1][i])-1-index] == 255:
                        b = (i,len(self.frames[1][i])-1-index)
            index += 1
        self.start = a
        self.end = b
        return

    def GetGrid(self):
        print()
        print("Creating Grid...")
        for key in self.frames.keys():
            Y = []
            frame = self.frames[key][:,self.start[1]:self.end[1]]
            x = 0
            while len(Y) < self.diff:
                count = 0
                for row in range(len(frame)):
                    if count ==0:
                        if frame[row][x] == 255:
                            val = self.start[0] - row
                            Y.append(val)
                            count+=1
                x+=1

            self.grid.append(Y)
        return

    def Constant(self):
        print()
        print("Finding Constants")
        for n in range(1, len(self.grid) - 1):
            for i in range(1, len(self.grid[n]) - 1):
                part2 = self.grid[n][i + 1] - 2 * self.grid[n][i]+ self.grid[n][i - 1]
                if part2 != 0 :
                    C2 = (self.grid[n + 1][i] + self.grid[n - 1][i] - 2 * self.grid[n][i] ) / part2
                else:
                    C2 = abs(self.grid[n + 1][i] + self.grid[n - 1][i]- 2 * self.grid[n][i])
                self.constants.append(C2)
        return







WaveEquation("movie.ogg",3)
