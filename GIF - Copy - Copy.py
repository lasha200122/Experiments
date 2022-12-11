from matplotlib.animation import PillowWriter
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
import matplotlib as mpl
import time, glob, os
import ffmpeg

res = []

# speed -> speed
# start -> start interval
# end -> end interval
# f -> function for u(x,0)
# g -> function for u_t(x,0)
# time -> time from 0 to T
# h -> Delta x
# k -> Delta t
# n -> number of intervals for X
# m -> number of intervals for t
# N -> number of iterations for analysis solution


# function for u(x,0)
def f(x):
    return np.sin(x)

# function for u_t(x,0)
def g(x):
    return 0

class String:
    def __init__(self, speed, start, end, f, g, time, n, m, N):
        self.speed = speed  # speed
        self.start = start  # start position
        self.end = end  # end position
        self.time = time  # time
        self.n = n  # number of intervals for x
        self.m = m  # number of intervals for time
        self.N = N  # number of iterations for analysis solution
        self.l = abs(self.end - self.start)  # length of string
        self.f = f  # function for u(x,0)
        self.g = g  # function for u_t(x,0 )

        self.h = abs(end - start) / n  # delta x
        self.k = abs(time) / m
        self.r = self.speed * self.k / self.h  # Constant that must be less 1

        self.domain = np.arange(self.start, self.end, self.h)  # domain for x axis interval [a,b]
        self.timeAxis = np.arange(0, self.time, self.k)  # list for time interval [0, time]

        self.gridForAnalysis = np.zeros((m, n))  # grid fo analysis method solution
        self.gridForNumerical = np.zeros((m, n))  # grid for numerical method solution
        self.difference = [] # List to calculate numerical and analysis solutions differences per frame
        #### Printing Part
        print("Speed: " + str(self.speed))
        print("Starting position: " + str(self.start))
        print("Ending position: " + str(self.end))
        print("Length of string: " + str(self.l))
        print("Time: " + str(self.time))
        print("Number of intervals for x: " + str(self.n))
        print("Number of intervals for time: " + str(self.m))
        print("delta x: " + str(self.h))
        print("delta t: " + str(self.k))
        print("Constant: " + str(self.r))
        ####

        if self.r >= 1:
            print()
            print("Can't solve this equation because Constant is: " + str(self.r) + "and it must be less than 1.")
        else:
            self.gridForAnalysis = self.AnalysisSolution()  # getting grid for Analysis solution
            self.gridForNumerical =-1 * self.NumericalSolution()  # getting grid for Numerical solution
            self.UpdateGridAnalysis()
            self.UpdateGridNumerical()
            self.SaveGif() # this gif contains analysis and numerical solutions
            # self.SaveAnalysisMp4(False) # this mp4 video contains only analysis solution
            # self.SaveNumericalMp4() # this mp4 video contains only numerical solution
            # self.SaveMP4() # this mp4 video contains analysis and numerical solutions
            # self.SaveAnalysisGif(False,False) # this gif contains only analysis solution
            # self.SaveNumericalGif() # this gif contains only numerical solution
            self.SaveDifferences() # this method saves png file of solution differences
            self.Error()
            print()
        print("Done.")

    def Error(self):
        difference = (self.gridForNumerical - self.gridForAnalysis)
        print(difference)
        # for i in range(len(difference)):
        #     for j in range(len(difference)):
        #         if self.gridForAnalysis[i][j] !=0:
        #             difference[i][j] = difference[i][j] / self.gridForAnalysis[i][j]
        # for i in difference:
        #     print(sum(i))
    def UpdateGridAnalysis(self):
        for i in range(len(self.gridForAnalysis)):
            self.gridForAnalysis[i][0] = 0
            self.gridForAnalysis[i][len(self.gridForAnalysis[i])-1] = 0
        self.gridForAnalysis[0] = self.f(self.domain)

    def UpdateGridNumerical(self):
        for i in range(len(self.gridForNumerical)):
            self.gridForNumerical[i][0] = 0
            self.gridForNumerical[i][len(self.gridForNumerical[i])-1] = 0

    def SinT(self, N, t):
        return np.sin(N * np.pi * self.speed * t / self.l)

    def CosT(self, N, t):
        return np.sin(N * np.pi * self.speed * t / self.l)

    def SinX(self, N):
        return np.sin(N * np.pi * self.domain / self.l)

    def NumecricalIntegrationForB(self, N):
        constant = 2 / self.l
        sum = 0
        for i in range(len(self.domain)):
            if i == 0 or i == len(self.domain) - 1:
                sum += self.f(self.domain[i]) * np.sin(N * np.pi * self.domain[i] / self.l)
            else:
                sum += 2 * self.f(self.domain[i]) * np.sin(N * np.pi * self.domain[i] / self.l)
        return sum * constant

    def NumericalIntegrationForBSharp(self, N):
        constant = 2 / (N * self.speed * np.pi)
        sum = 0
        for i in range(len(self.domain)):
            if i == 0 or i == len(self.domain) - 1:
                sum += self.g(self.domain[i]) * np.sin(N * np.pi * self.domain[i] / self.l)
            else:
                sum += 2* self.g(self.domain[i]) * np.sin(N * np.pi * self.domain[i] / self.l)
        return sum * constant

    def SummationForAnalysis(self, N, t):
        timePart = self.NumecricalIntegrationForB(N) * self.CosT(N, t) + self.NumericalIntegrationForBSharp(N) * self.SinT(N, t)
        result = timePart * self.SinX(N)
        return result

    def AnalysisSolution(self):
        if (self.N < 1):
            print("Analysis solution for this equation can't be solved because N = " + str(self.N))
            return
        grid = np.zeros((self.m, self.n))
        for i, t in enumerate(self.timeAxis):
            u = []
            for n in range(1, self.N + 1):
                if len(u) == 0:
                    u = np.array(self.SummationForAnalysis(n, t))
                else:
                    u += np.array(self.SummationForAnalysis(n, t))
            grid[i] = u
        print()
        print("Analysis solution have been found. Grid size:" + str(grid.shape))
        return grid

    def NumericalSolution(self):
        grid = np.zeros((self.m, self.n))
        grid[0] = self.f(self.domain)
        for j in range(1, len(grid) - 1):
            for i in range(1, len(grid[j]) - 1):
                if j == 1:
                    grid[1][i] = grid[0][i] + g(self.domain[i]) * self.k
                else:
                    grid[j + 1][i] = (2 - 2 * self.r ** 2) * grid[j][i] + np.power(self.r, 2) * (grid[j][i + 1] + grid[j][i - 1]) - grid[j - 1][i]
        print("Numerical solution have been found. Grid size:" + str(grid.shape))
        return grid

    def ErrorPerFrame(self):
        for iter in range(len(self.timeAxis)):
            print("Time: " + str(iter) + "solution numerical" + self.gridForNumerical[iter])
            print("Analysis" + self.gridForAnalysis[iter])

    def differenceGrid(self):
        print()
        print("Calculating errors...")
        difference = self.gridForNumerical - self.gridForAnalysis
        for frame in difference:
            value = abs(sum(frame))
            self.difference.append(value)
        print("Calculation Done")
        return

    def SaveDifferences(self):
        self.differenceGrid()
        fig = plt.figure()
        frames = np.arange(1,len(self.difference)+1,1)
        line = plt.plot(frames,self.difference, '--',color='red')
        plt.savefig("differences.png")
        return

    def SaveGif(self,border = True, number = True):
        metadata = dict(title="WaveEquation", artist="GroupWork")
        writer = PillowWriter(fps=15, metadata=metadata)
        maximum1 = max(map(max, self.gridForAnalysis))
        minimum1 = min(map(min, self.gridForAnalysis))
        maximum2 = max(map(max, self.gridForNumerical))
        minimum2 = min(map(min, self.gridForNumerical))
        maximum = max(maximum2,maximum1) + 1
        minimum = min(minimum2,minimum1) - 1
        fig = plt.figure(frameon=border)
        if not number:
            plt.xticks([])
            plt.yticks([])
        graph, = plt.plot([], [], label = "Analysis")
        graph2, = plt.plot([],[], label = "Numerical")
        plt.legend()
        plt.xlim(self.start-1, self.end+1)
        plt.ylim(minimum,maximum)
        with writer.saving(fig, "WaveEquation.gif", 100):
            for i in range(len(self.gridForNumerical)):
                graph2.set_data(self.domain,self.gridForNumerical[i])
                graph.set_data(self.domain, self.gridForAnalysis[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def SaveAnalysisGif(self,border=True, number=True):
        print()
        print("Saving Gif for analysis solution")
        metadata = dict(title="WaveAnalysisEquation", artist="GroupWork")
        writer = PillowWriter(fps=15, metadata=metadata)
        fig = plt.figure(frameon=border)
        graph, = plt.plot([], [])
        if not number:
            plt.xticks([])
            plt.yticks([])
        maximum = max(map(max,self.gridForAnalysis)) +1
        minimum = min(map(min, self.gridForAnalysis)) -1
        plt.xlim(self.start-1, self.end+1)
        plt.ylim(minimum, maximum)
        with writer.saving(fig, "WaveAnalysisEquation.gif", 100):
            for i in range(len(self.gridForAnalysis)):
                graph.set_data(self.domain, self.gridForAnalysis[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def SaveNumericalGif(self, border=True, number=True):
        print()
        print("Saving Gif for numerical solution")
        metadata = dict(title="WaveNumericalEquation", artist="GroupWork")
        writer = PillowWriter(fps=15, metadata=metadata)
        fig = plt.figure(frameon=border)
        if not number:
            plt.xticks([])
            plt.yticks([])
        graph, = plt.plot([], [])
        maximum = max(map(max,self.gridForNumerical)) +1
        minimum = min(map(min, self.gridForNumerical)) -1
        plt.xlim(self.start-1, self.end+1)
        plt.ylim(minimum, maximum)
        with writer.saving(fig, "WaveNumericalEquation.gif", 100):
            for i in range(len(self.gridForNumerical)):
                graph.set_data(self.domain, self.gridForNumerical[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def SaveAnalysisMp4(self,visible=True):
        print()
        print("Saving Analysis Video:")
        fig, ax = plt.subplots()
        if not visible:
            fig.patch.set_visible(False)
            ax.axis('off')
        ax.set_xlim(self.start-1,self.end+1)
        maximum = max(map(max, self.gridForAnalysis)) +1
        minimum = min(map(min,self.gridForAnalysis)) -1
        ax.set_ylim(minimum,maximum)
        line, = ax.plot(0,0)
        print()
        print("X axis -> start: " + str(self.start-1) + " | end: " + str(self.end +1))
        print("Y axis -> start: " + str(minimum) + " | end: " + str(maximum))
        print()
        print("Rendering Video...")
        print("In process...")
        def animation_frame(i):
            line.set_xdata(self.domain)
            line.set_ydata(self.gridForAnalysis[i])
            return line,
        animation = FuncAnimation(fig,func=animation_frame, frames=len(self.gridForAnalysis), interval=10)
        Writer = writers['ffmpeg']
        writer = Writer(fps=15,metadata={"artist":"Me"},bitrate=1800)
        animation.save("WaveAnalysisVideo.mp4",writer)
        print()
        print("Video Rendered Successfully")
        return



    def SaveNumericalMp4(self, visible=True):
        print()
        print("Saving Numerical Video:")
        fig, ax = plt.subplots()
        if not visible:
            fig.patch.set_visible(False)
            ax.axis('off')
        ax.set_xlim(self.start-1,self.end+1)
        maximum = max(map(max, self.gridForNumerical)) +1
        minimum = min(map(min,self.gridForNumerical)) -1
        ax.set_ylim(minimum,maximum)
        line, = ax.plot(0,0)
        print()
        print("X axis -> start: " + str(self.start-1) + " | end: " + str(self.end +1))
        print("Y axis -> start: " + str(minimum) + " | end: " + str(maximum))
        print()
        print("Rendering Video...")
        print("In process...")
        def animation_frame(i):
            line.set_xdata(self.domain)
            line.set_ydata(self.gridForNumerical[i])
            return line,
        animation = FuncAnimation(fig,func=animation_frame, frames=len(self.gridForNumerical), interval=10)
        Writer = writers['ffmpeg']
        writer = Writer(fps=15,metadata={"artist":"Me"},bitrate=1800)
        animation.save("WaveNumericalVideo.mp4",writer)
        print()
        print("Video Rendered Successfully")
        return

    def SaveMP4(self, visible=True):
        print()
        print("Saving Combination Video:")
        fig, ax = plt.subplots()
        if not visible:
            fig.patch.set_visible(False)
            ax.axis('off')
        ax.set_xlim(self.start-1,self.end+1)
        maximum1 = max(map(max, self.gridForNumerical)) +1
        minimum1 = min(map(min,self.gridForNumerical)) -1
        maximum2 = max(map(max, self.gridForAnalysis)) +1
        minimum2 = min(map(min,self.gridForAnalysis)) -1
        minimum = min(minimum2,minimum1)
        maximum = max(maximum2,maximum1)
        ax.set_ylim(minimum,maximum)
        line, = ax.plot(0,0, label="Numerical")
        line2, = ax.plot(0,0, label="Analysis")
        print()
        print("X axis -> start: " + str(self.start-1) + " | end: " + str(self.end +1))
        print("Y axis -> start: " + str(minimum) + " | end: " + str(maximum))
        print()
        print("Rendering Video...")
        print("In process...")
        def animation_frame(i):
            line.set_xdata(self.domain)
            line.set_ydata(self.gridForNumerical[i])
            line2.set_xdata(self.domain)
            line2.set_ydata(self.gridForAnalysis[i])
            return line, line2,
        animation = FuncAnimation(fig,func=animation_frame, frames=len(self.gridForNumerical), interval=10)
        Writer = writers['ffmpeg']
        writer = Writer(fps=15,metadata={"artist":"Me"},bitrate=1800)
        animation.save("WaveVideo.mp4",writer)
        print()
        print("Video Rendered Successfully")
        return

# Speed, starting point, ending point, function f , function g, Time, number of x intervals, number of time intervals, N
# String(1, 0, 2*np.pi, f, g, 5, 100, 300, 20)



# class Numerical:
#     def __init__(self,I,V,f,c,L,dt,C,T,umin,umax,user_action=None):
#         self.grid = []
#         self.T = T
#         self.viz(I, V,f, c, L, dt, C, T, umin, umax, animate=True)
#
#
#
#
#
#
#     def test_quadratic(self):
#         def u_exact(x, t):
#             return x * (L - x) * (1 + 0.5 * t)
#
#         def I(x):
#             return u_exact(x, 0)
#
#         def V(x):
#             return 0.5 * u_exact(x, 0)
#
#         def f(x, t):
#             return 2 * (1 + 0.5 * t) * c ** 2
#
#         L = 2.5
#         c = 1.5
#         C = 0.75
#         Nx = 30
#         dt = C * (L / Nx) / c
#         T = 18
#
#         def assert_no_error(u, x, t, n):
#             u_e = u_exact(x, t[n])
#             diff = np.abs(u - u_e).max()
#             tol = 1E-13
#             assert diff < tol
#
#         a, b, c = self.solver(I, V, f, c, L, dt, C, T, user_action=assert_no_error)
#
#     def guitar(self,C):
#         L = np.pi * 2
#         x0 = 0.8 * L
#         a = 0.005
#         freq = 440
#         wavelength = 2 * L
#         c = freq * wavelength
#         omega = 2 * np.pi * freq
#         num_periods = 1
#         T = 2 * np.pi / omega * num_periods
#         dt = L / 50. / c
#
#         def I(x):
#             return np.sin(x)  # a*x/x0 if x< x0 else a/(L -x0)*(L-x)
#
#         umin = -1.2 * a; umax = -umin
#         cpu = self.viz(I, 0, 0, c, L, dt, C, T, -5, 5, animate=True)
#

def I(x):
    return np.sin(x)

Data = []

def solver(I, V, f, c, L, dt, C, T, user_action=None):
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / float(C)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    C2 = C ** 2
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
    coeff = []
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + 0.5 * dt ** 2 * f(x[i],t[n])
    u[0] = 0; u[Nx] = 0

    if user_action is not None:
        user_action(u, x, t, 1)
    u_nm1[:] = u_n; u_n[:] = u
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt ** 2 * f(x[i], t[n])
            value = (u[i] + u_nm1[i] - 2 * u_n[i] - dt ** 2 * f(x[i], t[n]))/(u_n[i - 1] - 2 * u_n[i] + u_n[i + 1])
            coeff.append(np.sqrt(value))
        u[0] = 0 ; u[Nx] = 0

        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break
        u_nm1[:] = u_n; u_n[:] = u

    return u, x, t


def viz(I, V, f, c, L, dt, C, T, umin, umax, animate=True, tool='matplotlib', solver_function=solver):
    def plot_u_st(u, x, t, n):
        plt.plot(x, u, 'r--', xlabel='x', ylabel='u', axis=[0, L, umin, umax], title='t=%f' % t[n], show=True)

        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig("frame_%04d.png" % n)

    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            res.append(list(u))
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u)
                plt.box(False)
                plt.axis('off')
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, L, umin, umax])
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
    u, x, t = solver_function(I, V, f, c, L, dt, C, T, user_action)
    fps = 100
    codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm', libtheora='ogg')
    filespec = 'tmp_%04d.png'
    movie_program = 'ffmpeg'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = 'C:/Users/KiuAdmin/Desktop/ffmpeg-5.1-full_build/bin/ffmpeg.exe -r %(fps)d -i %(filespec)s -vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)
    return

def I0(x):
    return np.sin(2*np.pi*x/3)

def V0(x):
    return np.sin(2*np.pi*x/3) * np.pi*2 / 3

def I2(x):
    return np.sin(x/2)

def I3(x):
    return np.sin(2*np.pi*x/3)

def V3(x):
    return np.sin(2*np.pi*x/3)*2*np.pi/3

def I4(x):
    return np.sin(2* np.pi*x/5) + 2* np.sin(3* np.pi * x/5)

def V4(x):
    return np.sin(2 * np.pi *x /5) * 3 * np.pi /5 + np.sin(3 *np.pi* x/5) * 6 *1.5*np.pi/5
result = viz(I4,V4,0,1.5,5,0.01,0.5,10,-5,5)
# print(result)
###############
# I -> u(x,0)
# V -> u_t(x,0)
# f -> u_tt = c^2 u_xx + f(x,t)
# c -> speed
# L -> Length
# dt -> delta t
# C -> constant | c dt / dx  | where dx -> delta x and c -> speed
# T -> time
# umin -> minimum height value that string approaches
# umax -> maximum height value that string approaches

def exact(A,x,c,L,t):
    return A * np.sin(np.pi*x/L) * np.cos(np.pi*c*t/L)

def Analysis(A,L,T,c,dt,C):
    Nt = int(round(T/dt))
    time = np.linspace(0,Nt*dt,Nt+1)
    dx = dt * c / float(C)
    Nx = int(round(L / dx))
    domain = np.linspace(0, L, Nx + 1)
    grid = []
    for t in time:
        ls = exact(A,domain,c,L,t)
        grid.append(ls)
    return grid, domain

def F(x,t):
    result = (np.cos(2*np.pi*t/3)+ np.sin(2*np.pi*t/3))*np.sin(2*np.pi*x/3)
    return result

def F2(x,t):
    result = (np.cos(3*np.pi*t/5) + np.sin(3*np.pi*t/5)) * np.sin(2*np.pi * x / 5) + (2 * np.cos(4.5*np.pi*t/5) + 2* np.sin(4.5*np.pi*t/5))* np.sin(3*np.pi*x/5)
    return result
def Test(T,dt,L,c,C,F):
    Nt = int(round(T / dt))
    time = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / float(C)
    Nx = int(round(L / dx))
    domain = np.linspace(0, L, Nx + 1)
    grid = []
    for t in time:
        ls = F(domain,t)
        grid.append(ls)
    return grid, domain

plt.close('all')
AnalysisGrid, domain =  Test(5,0.01,5,1.5,0.5,F2) # Analysis(1,3,5,1,0.1,0.5)


print()
print("Saving Gif for analysis solution")
metadata = dict(title="WaveAnalysisEquation", artist="GroupWork")
writer = PillowWriter(fps=15, metadata=metadata)
fig = plt.figure()
graph, = plt.plot([], [], label="Real")
graph2, = plt.plot([],[], label="Numerical")
plt.legend()
maximum = max(map(max,AnalysisGrid)) +1
minimum = min(map(min, AnalysisGrid)) -1
plt.xlim(0, 7)
plt.ylim(minimum, maximum)
print(result)
with writer.saving(fig, "WaveAnalysisEquation222.gif", 100):
    for i in range(len(AnalysisGrid)):
        graph.set_data(domain, AnalysisGrid[i])
        graph2.set_data(domain,res[i])
        writer.grab_frame()
print()
print("Gif Saved Successfully")


difference = np.array(AnalysisGrid) - np.array(res)
y = []
x = []
for j in range(len(AnalysisGrid)):
    diff = AnalysisGrid[j] - res[j]
    y.append(abs(sum(diff)))
    x.append(j)


plt.close('all')
plt.figure()
plt.plot(x,y,color='red')
plt.savefig("Error.png")
dx = domain[1] - domain[0]
y2 = []
for j in range(len(AnalysisGrid)):
    diff = AnalysisGrid[j] - res[j]
    val = np.sqrt(sum([i**2 for i in diff])*dx)
    y2.append(val)

plt.figure()
plt.plot(x,y2,color='red')
plt.savefig("ErrorL2.png")

for filename in glob.glob("tmp_*.png"):
    os.remove(filename)

coeff = []
for n in range(1,len(res)-1):
    for i in range(1,len(res[n])-1):
        C2 = (res[n+1][i] + res[n-1][i] - 2* res[n][i])/ (res[n][i+1] - 2* res[n][i] + res[n][i-1])
        value = np.sqrt(C2)
        coeff.append(value)



# dt -> 0.5

