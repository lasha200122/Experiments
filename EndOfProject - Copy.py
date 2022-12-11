import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import cv2
import time
import glob
import os
import imageio
from scipy.interpolate import interp1d

'''
Hello this python code is written for Wave Equations
Here is some good methods that you can use or modify 
for your purposes. using this methods you can create 
gif and video visualisation for your wave equation 
problem. also you can determine moving string's wave
equation using video. I think there is really interesting
problems to discuss and modify.

There also can occur some errors. The method for saving
video will not work if you do not have ffmpeg.exe in your
computer. you can download it from this link: https://ffmpeg.org/download.html
do not forget to input the path of ffmpeg.exe in the input fields later ;)

For better understanding this algorithms let's talk about annotations.
Our Wave Equation -> u_tt = c^2 u_xx + f(x,t) 
this methods is considered for boundary, initial value problems so there
we will have: u(0,t) , u(L,t) , u(x,0), u_t(x,0) and also f(x,t)

****  WaveSolver attributes ****
I -> Function for u(x,0)
V -> Function for u_t(x,0)
f -> Function for f(x,t)
c -> Speed of wave
L -> Length of wave
dt -> Delta t
dx -> Delta x
Cfl -> Constant (computation Cfl = c * dt / dx  )
T -> Time (duration)
Amax -> maximum amplitude of given wave equation
ffmpeg -> path of ffmpeg.exe
Func -> another function for wave equation
path -> video path 


Solver method solves wave equation using finite differences.

viz method visualizes solution of wave equation.

RemovePictures method removes all pictures that where stored while using viz method (nothing important just fast way to 
delete junked files).

ExactEquation method is formula u_e(x,t) = A sin(pi x / L) cos(pi c t / L) and returns it's value at the current t and 
domain.

Analysis method returns analytical solution of wave equation. (mostly this is used to check correctness of numerical 
solutions).

WithMyFunction method return also analytical solution of wave equation. as you know there are too many ways to solve
wave equation so this method returns solutions of separation variables 

NumericalGif method saves numerical solution as gif

AnalysisGif method saves analysis solution as gif

CompareGif method saves analysis and numerical solutions as one gif

PlotError method calculates error between numerical and analysis solutions and then plots it as graph and saves it as 
png.
'''

"Let's Create input attributes and functions"


def I(x):
    return np.sin(2*np.pi*x/3)


def V(x):
    return np.sin(2*np.pi*x/3) * np.pi*2 / 3


def Func(x,t):
    return (np.cos(2*np.pi*t/3) + np.sin(2*np.pi*t/3)) * np.sin(2 * np.pi * x / 3)


f = 0
c = 1
L = 3
dt = 0.01
Cfl = 0.5
T = 10
Amax = 3
ffmpeg = 'C:/Users/KiuAdmin/Desktop/ffmpeg-5.1-full_build/bin/ffmpeg.exe'
video_path = 'C:/Users/KiuAdmin/Desktop/String Vibration/Test 1/movie.ogg'

## Method for solving Wave Equation

def solver(I, V, f, c, L, dt, Cfl, T, user_action=None):
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / float(Cfl)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    C2 = Cfl ** 2
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if f is None or f == 0:
        f = lambda x, t: 0
        if V is None or V == 0:
            V = lambda x: 0
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)
    u_nm1 = np.zeros(Nx + 1)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    n = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + 0.5 * dt ** 2 * f(x[i],t[n])
    u[0] = 0; u[Nx] = 0
    if user_action is not None:
        user_action(u, x, t, 1)
    u_nm1[:] = u_n; u_n[:] = u
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt ** 2 * f(x[i], t[n])
        u[0] = 0; u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break
        u_nm1[:] = u_n; u_n[:] = u
    return u, x, t

#Method for visualizing Wave Equation

def viz(I, V, f, c, L, dt, Cfl, T, animate=True, solver_function=solver):
    def plot_u_st(u, x, t, n):
        plt.plot(x, u, 'r--', xlabel='x', ylabel='u', axis=[0, L, -Amax, Amax], title='t=%f' % t[n], show=True)
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig("frame_%04d.png" % n)
    result = []
    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            result.append(list(u))
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u)
                # you can also turn border and axis on.
                plt.box(False)
                plt.axis('off')
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, L, -Amax, Amax])
                # you can add this for timer
                # plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                # plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
            time.sleep(2) if t[n] == 0 else time.sleep(0.2)
            plt.savefig('tmp_%04d.png' % n)
    plot_u = PlotMatplotlib()
    for filename in glob.glob("tmp_*.png"):
        os.remove(filename)
    user_action = plot_u if animate else None
    u, x, t = solver_function(I, V, f, c, L, dt, Cfl, T, user_action)
    fps = int(1/dt)
    codec2ext = dict(libtheora='ogg')
    filespec = 'tmp_%04d.png'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = ffmpeg+' -r %(fps)d -i %(filespec)s -vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)
    return result

class WaveSolver:
    def __init__(self, I, V, f, c, L, dt, Cfl, T, Amax, ffmpeg, Func = None ,user_action=None):
        self.I = I
        self.V = V
        self.f = f
        self.c = c
        self.L = L
        self.dt = dt
        self.Cfl = Cfl
        self.T = T
        self.fps = int(1 / self.dt)
        self.Amax = Amax
        self.ffmpeg = ffmpeg
        self.Func = Func
        self.user_action = user_action
        self.Nt = int(round(self.T / self.dt))
        self.time = np.linspace(0, self.Nt * self.dt, self.Nt + 1)
        self.dx = self.dt * self.c / float(self.Cfl)
        self.Nx = int(round(self.L / self.dx))
        self.domain = np.linspace(0, self.L, self.Nx + 1)

        ## Solving Part
        self.numericalGrid = viz(self.I, self.V, self.f, self.c, self.L, self.dt, self.Cfl, self.T)
        if self.Func == None:
            self.analysisGrid = self.Analysis()
        else:
            self.analysisGrid = self.WithMyFunction()

        ### Saving Part
        self.AnalysisGif()
        self.NumericalGif()
        self.CompareGif()
        self.PlotError()

        ### Printing Part
        print()
        print("Speed: " + str(self.c))
        print("Length: " + str(self.L))
        print("Delta x: " + str(self.dx))
        print("Delta t: " + str(self.dt))
        print("Time: " + str(self.T))
        print("Constant: " + str(self.Cfl))
        print("Amplitude: " + str(self.Amax))
        print("Time intervals: " + str(self.Nt))
        print("Domain intervals: " + str(self.Nx))
        print("FPS: " + str(self.fps))
        print()

    def RemovePictures(self):
        for filename in glob.glob("tmp_*.png"):
            os.remove(filename)
        return True

    def ExactEquation(self, A, x, c, L, t):
        return A * np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    def Analysis(self):
        grid = []
        for t in self.time:
            ls = self.ExactEquation(self.Amax, self.domain, self.c, self.L, t)
            grid.append(ls)
        return grid

    def WithMyFunction(self):
        grid = []
        for t in self.time:
            ls = self.Func(self.domain, t)
            grid.append(ls)
        return grid

    def NumericalGif(self):
        print()
        print("Saving Numerical Gif")
        metadata = dict(title="NumericalSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Numerical.gif", 100):
            for i in self.numericalGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def AnalysisGif(self):
        print()
        print("Saving Analysis Gif")
        metadata = dict(title="AnalysisSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Analysis.gif", 100):
            for i in self.analysisGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def CompareGif(self):
        print()
        print("Saving Compare Gif")
        metadata = dict(title="Compare", artist="Lasha_Shavgulidze")
        fps = int(1 / self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        graph2, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Compare.gif", 100):
            for i in range(len(self.numericalGrid)):
                graph.set_data(self.domain, self.numericalGrid[i])
                graph2.set_data(self.domain, self.analysisGrid[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def PlotError(self):
        y = []
        x = []
        for j in range(len(self.analysisGrid)):
            diff = self.analysisGrid[j] - self.numericalGrid[j]
            val = np.sqrt(sum([i ** 2 for i in diff]) * self.dx)
            y.append(val)
            x.append(j)
        plt.figure()
        plt.plot(x, y, color='red')
        plt.savefig("Error.png")
        return

class WaveFinder:
    def __init__(self, path, L, n, clean=True):
        self.path = path
        self.n = n
        self.L = L
        self.frames = {}
        self.fps = 0
        self.frame_count = 0
        self.dt = 0
        self.dx = 0
        self.time = 0
        self.shape = (0, 0)
        self.start = (0, 0)
        self.end = (0, 0)
        self.domain = np.arange(0, self.L, self.L / self.n)
        self.solution = []
        self.scale = 0
        self.difference = 0
        self.constants = []

        if clean:
            self.FindCleanedValues()
        else:
            self.FindNoisedValues()
        # self.AnalysisGif()
        self.FindPoints()
        self.GetGrid()
        self.scatter()
        # print(len(self.solution))
        # self.Solve()

        # self.FindValues()
        # self.FindPoints()
        # self.Constant()


    def FindCleanedValues(self):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = self.frame_count / self.fps
        self.dt = self.time / self.frame_count
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        print("Getting Data")
        print("Frame count: " + str(self.frame_count))
        count = 0
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=33, sigmaY=33)
                edges = cv2.divide(frame, blur, scale=255)
                edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                edges = 255 - edges
                if count < 1:
                    print("Shape of frame: " + str(edges.shape))
                    print("Reading frames...")
                    self.shape = edges.shape

                edges[edges != 255] = 0
                self.frames[count] = edges
                cv2.imwrite("frame" + str(count) + ".png", edges)
                # cv2.imshow("video",edges)
                count+=1
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        print()
        print("Finished reading frames")
        return

    def FindNoisedValues(self):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = self.frame_count / self.fps
        self.dt = self.time / self.frame_count
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        print("Getting Data")
        print("Frame count: " + str(self.frame_count))
        count = 0
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.medianBlur(frame,3)
                edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
                edges = 255- edges
                if count < 1:
                    print("Shape of frame: " + str(edges.shape))
                    print("Reading frames...")
                    self.shape = edges.shape
                count += 1
                edges[edges != 255] = 0
                self.frames[count] = edges
                cv2.imshow("video",edges)
                cv2.imwrite("frame" + str(count) + ".png", edges)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        print()
        print("Finished reading frames")
        return

    def FindPoints(self):
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
        self.difference = abs(a[1] - b[1])
        self.scale = self.L / self.difference

        return

    def AnalysisGif(self):
        ls = self.frames.values()
        with imageio.get_writer("smiling.gif", mode="I") as writer:
            for idx, frame in enumerate(ls):
                print("Adding frame to GIF file: ", idx + 1)
                writer.append_data(frame)
        return

    def GetGrid(self):
        print()
        print("Creating Grid...")
        step = abs(self.start[1] - self.end[1]) // self.n
        for key in self.frames.keys():
            Y = []
            X = []
            coordinates = []
            frame = self.frames[key][:,self.start[1]:self.end[1]+1]
            x = 0

            while x < len(frame[0]):
                count = 0
                ls = []
                for i in range(len(frame)):
                    if count == 0:
                        if frame[i][x] == 255:

                            ls.append(self.start[0] - i)
                    #//count+=1
                Y.append(np.median(ls))
                X.append(x)
                x+=1
            for i in range(len(X) - 1):
                for j in range(i,len(X)):
                    if X[i] > X[j]:
                        X[i], X[j] = X[j], X[i]
                        Y[i], Y[j] = Y[j], Y[i]
            f = interp1d(X,np.array(Y)* self.scale, kind='linear')
            newY = f(X)
            plt.plot(X,newY)
            # plt.show()
            self.solution.append((X,Y))
            self.dx = self.L / len(Y)
        return


    def scatter(self):
        for i in self.solution:
            plt.scatter(i[0],i[1], s=0.1)
            plt.show()

    def Solve(self):
        print(self.solution)
        for i in range(2,len(self.solution)):
            ls = []
            for j in range(2, len(self.solution[i][1])-1):
                part = self.solution[i-1][1][j] - self.solution[i-1][1][j-1] + self.solution[i-1][1][j-2]
                if part != 0:
                    C = (self.solution[i][1][j-1] + self.solution[i-2][1][j-1] - 2* self.solution[i-1][1][j-1]) / part
                    ls.append(C)
            print(ls)
            print(max(ls))



WaveFinder(video_path,L,150)