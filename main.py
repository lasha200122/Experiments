import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import cv2

class SecantMethod:
    def __init__(self, FirstGuess, SecondGuess, From, To, Step,error):
        #error
        self.error = error
        #Domain of X
        self.X = np.arange(From,To,Step)
        #Range of Y
        self.Y = self.Function(self.X)

        #Plot Graph
        self.PlotGraph(self.X,self.Y)

        #FindRoot f(x) = 0
        print(self.FindRoot(FirstGuess,SecondGuess))

    #Main Function
    def Function(self,x):
        Y = 0.5 * x**3 + 4* x**2 -18*x + 20
        return Y

    #Ploting Graph
    def PlotGraph(self,X,Y):
        plt.plot(X,Y)
        plt.show()

    #Ploting Grap with Line to show how algorithm works
    def DrawLine(self,x1,x2):
        plt.plot(self.X,self.Function(self.X))
        Y = [self.Function(x1),self.Function(x2)]
        X = [x1,x2]
        plt.plot(X,Y)
        plt.show()

    #Function for Algorithm
    def FindRoot(self, x1,x2):
        root = 0
        while np.abs(x2-x1) > self.error:
            self.DrawLine(x1,x2)
            root = x2 - self.Function(x2)/((self.Function(x2) - self.Function(x1))/(x2-x1))
            common = x2
            x2 = root
            x1 = common
        return x2

#Guess, Start, End,Step, error
# SecantMethod(-14,-10,-20,10,0.01,0.00000000001)

class NewthonMethod:
    def __init__(self,Guess,Start,End,Step,error):
        #Main part
        self.error = error
        self.Start = Start
        self.End = End
        self.Step = Step
        self.ls = []
        #Graph
        self.PlotGraph(np.arange(Start,End,Step),self.Function(np.arange(Start,End,Step)))

        #Solution
        print(self.FindRoot(Guess,0))


    #Mainc Function
    def Function(self,x):
        Y = 0.5 * x ** 3 + 4 * x ** 2 - 18 * x + 20
        return Y

    #Derivative Calcualtion Returns Float
    def DerivativeFunction(self,z):
        x = Symbol('x')
        y = 0.5 * x ** 3 + 4 * x ** 2 - 18 * x + 20
        dx = diff(y, x)
        return dx.evalf(z, subs={x: z})

    #Ploting Graph
    def PlotGraph(self,X,Y):
        plt.plot(X,Y)
        return plt.show()

    #Ploting Graph With Tangent Line
    def PlotTangentLine(self,x):
        X = np.arange(self.Start,self.End,self.Step)
        Y = self.Function(X)
        plt.plot(X,Y)
        plt.plot([-20,20],[self.TangentLine(-20,x),self.TangentLine(20,x)])
        plt.scatter(self.ls, [self.Function(i) for i in self.ls])
        return plt.show()

    #Tangent Line Formula
    def TangentLine(self,z,x):
        return self.DerivativeFunction(x) *(z - x) +self.Function(x)

    #Main Function for Algorithm
    def FindRoot(self, x,check):
        if (abs(x - check) < self.error):
            return x
        if self.DerivativeFunction(x) != 0:
            check = x - self.Function(x)/self.DerivativeFunction(x)

            self.ls.append(x)
            self.PlotTangentLine(x)
            return self.FindRoot(check,x)
        return "No solution for this guess"

#Guess, Start, End,Step, error
NewthonMethod(-20,-20,20,0.1,0.00000000001)

class FixedPointIteration:
    def __init__(self,Guess,Start,End,Step,error):
        self.error = error

        if (self.Check):
            self.PlotGraph(np.arange(Start,End,Step),self.FunctionForGraph(np.arange(Start,End,Step)))
            print(self.RootFinder(Guess,0))
        else:
            print("The method does not converges in this case")

    #Plotting
    def PlotGraph(self,x,y):
        plt.plot(x,y)
        return plt.show()

    #For Plotting
    def FunctionForGraph(self,x):
        return x**2 - x - 1

    #We took x**2 - x -10 Function
    def Function(self,x):
        return 1 + 1/x

    #Finding Derivative Of Function on Current point
    def DerivativeFunction(self,z):
        x = Symbol('x')
        y = 1 + 1/x
        dx = diff(y, x)
        return dx.evalf(z, subs={x: z})

    #Check if Algorithm converges for this equation
    def Check(self,x):
        if abs(self.DerivativeFunction(x))<=1:
            return True
        return False

    #Main Function for algorithm
    def RootFinder(self,x,root):
        if (abs(x-root)<self.error):
            return x
        return self.RootFinder(1+1/x,x)

#Guess,Start,End,Step,Error
# FixedPointIteration(5,-5,5,0.1,0.00000000001)

class RegulaFalsiMethod:
    def __init__(self,Start,End,error):
        #Info
        self.error = error

        #Check If method can solve the equation
        if self.Check(Start,End):
            #plot Graph
            self.PlotGraph(np.arange(Start-10,End+10,0.1),self.Function(np.arange(Start-10,End+10,0.1)))
            print(self.RootFinder(Start,End,-2))
        else:
            print("This method does not work for this interval")

    #Main equation
    def Function(self,x):
        Y = x**2 - x - 1
        return Y

    #Function For Plotting
    def PlotGraph(self,x,y):
        plt.plot(x,y)
        return plt.show()

    #Checker Function for solution
    def Check(self,a,b):
        if self.Function(a)*self.Function(b) < 1:
            return True
        return False

    #Main Function For Algorithm
    def RootFinder(self,a,b,c):
        if self.Function(c) == 0 or abs(self.Function(c)) < self.error:
            return c
        k = 1
        if (self.Function(b) - self.Function(a) !=0):
            k = b - self.Function(b) *(b-a)/(self.Function(b) - self.Function(a))

        if (self.Function(c)*self.Function(k) > 0):
            return self.RootFinder(k,b,k)

        return self.RootFinder(a,k,k)

# RegulaFalsiMethod(-5,0,0.00000000001)
# RegulaFalsiMethod(1,4,0.00000000001)

class ChordMethod:
    def __init__(self,Start,End,error):
        self.error= error
        if self.Check(Start,End):
            print(self.RootFinder(Start,Start,End))
        else:
            print("This Method can't solve the equation for this interval")

    def Function(self,x):
        y = x**2 - x - 1
        return y

    def Check(self,a,b):
        if self.Function(a)*self.Function(b) < 0:
            return True
        return False

    def RootFinder(self,x,a,b):
        if abs(self.Function(x)) < self.error:
            return x
        return self.RootFinder(x - (self.Function(x)*(b-a))/(self.Function(b)-self.Function(a)),a,b)

# ChordMethod(-5,0,0.00000000001)
# ChordMethod(1,4,0.00000000001)

class UndeterminedCoefficients1DFirstOrder:
    def __init__(self, Start, End, FStart, FEnd, N, Stencil):
        self.x = np.arange(Start, End + (End - Start) / N, (End - Start) / N)
        self.A = np.array(self.UpdateA(np.array(np.matrix(np.zeros((N + 1, N + 1)))), N + 1, Stencil))
        self.B = np.array(self.UpdateB(np.array(np.matrix(np.zeros((N + 1, 1)))), N + 1, FStart, FEnd))
        self.y = [i[0] for i in np.linalg.solve(self.A, self.B)]
        self.PlotGraph(self.x, self.y)

    def PlotGraph(self, x, y):
        if (len(x) != len(y)):
            x = x[:-1]
        plt.scatter(x, y)
        return plt.show()

    def UpdateA(self, A, N, Stencil):
        A[0][0] = 1
        A[N - 1][N - 1] = 1
        for i in range(1, N - 1):
            A[i][i - 1] = Stencil[0]
            A[i][i] = Stencil[1]
        return A

    def UpdateB(self, B, N, FStart, FEnd):
        B[0] = FStart
        B[N - 1] = FEnd
        for i in range(1, N - 1):
            B[i] = self.F(self.x[i])
        return B

    def F(self, x):
        return np.sin(x)

# UndeterminedCoefficients1DFirstOrder(np.pi,np.pi*2,1,0,30,[1-30/np.pi, 30/np.pi])

class UndeterminedCoefficients1DSecondOrder:
    def __init__(self,Start,End,FStart,FEnd,N,Stencil):
        self.x = np.arange(Start,End+(End-Start)/N,(End-Start)/N)
        self.A = np.array(self.UpdateA(np.array(np.matrix(np.zeros((N+1, N+1)))),N+1,Stencil))
        self.B = np.array(self.UpdateB(np.array(np.matrix(np.zeros((N+1,1)))),N+1,FStart, FEnd))
        self.y = [i[0] for i in np.linalg.solve(self.A,self.B)]
        self.PlotGraph(self.x,self.y)

    def PlotGraph(self,x,y):
        if (len(x)!= len(y)):
            x = x[:-1]
        plt.scatter(x,y)
        return plt.show()

    def UpdateA(self,A,N,Stencil):
        A[0][0] = 1
        A[N - 1][N - 1] = 1
        for i in range(1,N-1):
            A[i][i-1] = Stencil[0]
            A[i][i] = Stencil[1]
            A[i][i+1] = Stencil[2]
        return A

    def UpdateB(self,B,N,FStart,FEnd):
        B[0] = FStart
        B[N-1] = FEnd
        for i in range(1,N-1):
            B[i]= self.F(self.x[i])
        return B

    def F(self,x):
        return 0

# UndeterminedCoefficients1DSecondOrder(0,np.pi,1,-1,20,[400/np.power(np.pi,2) + 20/np.pi , 2 - 800/np.power(np.pi,2),400/np.power(np.pi,2) - 20/np.pi])

class UndeterminedCoefficients2DFirstOrder:
    def __init__(self,a,b,nx,ny,iteration):
        self.dx = a / nx
        self.dy = b / ny
        self.u = np.array(np.zeros((ny + 1, nx + 1)))
        self.x = np.arange(0, a + self.dx, self.dx)
        self.y = np.arange(0, b + self.dy, self.dy)
        self.u = self.UpdateU(self.dx,self.dy,nx,ny,iteration)
        self.PlotGprah(self.x,self.y,self.u)

    def UpdateU(self,dx,dy,nx,ny,iteration):
        self.u[0, :] = 0
        self.u[:, 0] = 0
        self.u[ny, :] = np.sin(self.x)
        self.u[:, nx] = np.cos(self.y)
        for k in range(iteration):
            for i in range(1,ny):
                for j in range(1, nx):
                    if (self.u[i,j] ==0.):
                        self.u[i,j] = self.u[i+1,j]/(dx**2 + dy) + self.u[i,j+1]/(dy**2 + dx)
        return self.u

    def PlotGprah(self,x,y,u):
        plt.contourf(x,y,u,30)
        return plt.show()

# UndeterminedCoefficients2DFirstOrder(5,10,100,50,5000)

class UndeterminedCoefficients2DSecondOrder:
    def __init__(self,a,b,nx,ny,iteration):
        self.dx = a / nx
        self.dy = b / ny
        self.u = np.array(np.zeros((ny+1,nx+1)))
        self.x = np.arange(0,a+ self.dx,self.dx)
        self.y = np.arange(0,b + self.dy,self.dy)
        self.UpdateU(self.dx,self.dy,nx,ny,a,b,iteration)
        self.PlotGprah(self.x,self.y,self.u)

    def PlotGprah(self,x,y,u):
        plt.contourf(x,y,u,30)
        return plt.show()

    def UpdateU(self,dx,dy,nx,ny,a,b,iteration):

        h = 1 / (2 / (dx**2 + dy**2))
        self.u[0,:] = 0
        self.u[ny,:] = np.sin(self.x) / np.sin(a)
        self.u[:,0] = 0
        self.u[:,nx] = np.sinh(self.y) / np.sinh(b)
        for k in range(iteration):
            for i in range(1,ny):
                for j in range(1, nx):
                    if (self.u[i,j] ==0.): self.u[i,j] = h * (dx**2 * (self.u[i+1,j] + self.u[i-1,j]) + dy**2 * (self.u[i,j+1] + self.u[i,j-1]))
        return self.u

# UndeterminedCoefficients2DSecondOrder(3*np.pi,np.pi,100,50,900)

class EdgeDetection:
    def __init__(self,path):
        self.MyImage(path)
        self.RobertsOperator(path)
        self.RobertsOperatorSecond(path)
        self.SobelOperator(path)
        self.Laplacian2DSecondOrder(path)
        self.Laplacian2DSecondOrder2ndFilter(path)
        self.Laplacian2DFirstOrder(path)
        self.D1SecondOrder(path)
        self.D1FirstCentral(path)
        self.D1FirstForward(path)

    def MyImage(self,path):
        img = np.array(cv2.imread(path))
        cv2.imshow('image', img)
        return cv2.waitKey(0)

    def RobertsOperator(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        for i in range(1,500-1):
            for j in range(1,500-1):
                img[i][j]= - int(img[i][j]) + int(img[i+1][j+1])
        cv2.imshow('image', img)
        return cv2.waitKey(0)

    def RobertsOperatorSecond(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        for i in range(1,500-1):
            for j in range(1,500-1):
                img[i][j]= - int(img[i][j]) + int(img[i+1][j-1])
        cv2.imshow('image', img)
        return cv2.waitKey(0)

    def D1SecondOrder(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500,500))

        for i in range(shape(img)[0]):
            for j in range(1,shape(img)[1]-1):
                q = -6*img[i][j] + img[i][j-1]*6 + img[i][j+1]*2
                if q> 15: #threshhold  200 180
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0

        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def SobelOperator(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500,500))

        for i in range(shape(img)[0]):
            for j in range(shape(img)[1]):
                q = (img[i][j]**2 + img[i][j]**2)**0.5
                if q> 90: #threshhold  200 180
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0

        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def Laplacian2DSecondOrder(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500, 500))
        for i in range(1,shape(img)[0]-1):
            for j in range(1,shape(img)[1]-1):
                q = 4*img[i][j] - img[i-1][j] - img[i+1][j] - img[i][j-1] -img[i][j+1]
                if q> 20:
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0
        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def Laplacian2DFirstOrder(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500, 500))
        for i in range(1,shape(img)[0]-1):
            for j in range(1,shape(img)[1]-1):
                q = -img[i][j] - 0.5 * img[i+1][j] + 0.5 * img[i][j+1]
                if q> 40:
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0
        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def Laplacian2DSecondOrder2ndFilter(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500, 500))
        for i in range(1,shape(img)[0]-1):
            for j in range(1,shape(img)[1]-1):
                q = 8*img[i][j] - img[i-1][j] - img[i+1][j] - img[i][j-1] -img[i][j+1] - img[i+1][j+1] - img[i-1][j+1] - img[i-1][j-1] - img[i+1][j-1]
                if q> 20: ## threshhold experiments
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0
        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def D1FirstCentral(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500,500))
        for i in range(1,499):
            for j in range(1,499):
                if  0.5*int(img[i][j-1]) - 0.5* int(img[i][j+1]) >20:
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0
        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

    def D1FirstForward(self,path):
        img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        newimg = np.zeros((500,500))
        for i in range(1,499):
            for j in range(1,499):
                if  int(img[i][j+1]) -  int(img[i][j]) >20:
                    newimg[i][j] = 255
                else:
                    newimg[i][j] = 0
        cv2.imshow('image', newimg)
        return cv2.waitKey(0)

# EdgeDetection("ss.png")

class GridSpacing:
    def __init__(self,a,b,c,h,start,end,step):
        self.n = np.arange(start,end,step)

        y1 = self.First(a,b,c,h)
        y2 = self.Second(a,b,c,h)
        y3 = self.Third(a,b,c,h)

        fig, axs = plt.subplots(3)
        fig.suptitle('Graphs')
        axs[0].plot(self.n, y1)
        axs[1].plot(self.n, y2)
        axs[2].plot(self.n,y3)
        plt.show()

    def X(self,v):
        return np.sin(v)

    def Dx(self,v):
        return np.cos(v)

    def DDx(self,v):
        return - np.sin(v)

    def xddd(self,v):
        return - np.cos(v)

    def First(self,a,b,c,h):
        return self.X(self.n) * (a[0] / h + b[0] + h * c[0] + a[1] / h + b[1] + h * c[1] + a[2] / h + b[2] + h * c[2]) + self.Dx(self.n) * (
                    a[1] + b[1] * h + c[1] * np.power(h, 2) - a[0] - b[0] * h - c[0] * np.power(h, 2))

    def Second(self,a,b,c,h):
        return self.X(self.n) * (a[0] / h + b[0] + h * c[0] + a[1] / h + b[1] + h * c[1] + a[2] / h + b[2] + h * c[2]) + self.Dx(self.n) * (a[1] + b[1] * h + c[1] * np.power(h, 2) - a[0] - b[0] * h - c[0] * np.power(h, 2))+ self.DDx(self.n) * (0.5*(h*a[2] + h * a[0] + np.power(h,2) * b[2] + np.power(h,2) * b[0] + np.power(h,3) *c[2] + np.power(h,3) * c[0]))

    def Third(self,a,b,c,h):
        return self.X(self.n) * (
                    a[0] / h + b[0] + h * c[0] + a[1] / h + b[1] + h * c[1] + a[2] / h + b[2] + h * c[2]) + self.Dx(
            self.n) * (a[1] + b[1] * h + c[1] * np.power(h, 2) - a[0] - b[0] * h - c[0] * np.power(h, 2)) + self.DDx(
            self.n) * (0.5 * (h * a[2] + h * a[0] + np.power(h, 2) * b[2] + np.power(h, 2) * b[0] + np.power(h, 3) * c[
            2] + np.power(h, 3) * c[0])) + self.xddd(self.n) * (np.power(h,3)*(a[2]-a[0]) + np.power(h,4)* (b[2]- b[0]) + np.power(h,6) * (c[2] - c[0]))/6

# GridSpacing([-1,0,1],[0,0,0],[0,0,0],0.5,-50,50,0.1)