import os
import imageio
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like

"""Hello To Run the program you have to create folders with Names:
    1) Triangle
    2) Rectangle
    3) Polygon
    4) Circle
    This Folders have to be located in the folder where .Py file is.
    !! Remember this names are unique string so try to copy them while creating this folders 

    This programs will save images in this folders and after that it will generate gif
    using this images. 

    If you do not want to save this images after program stops working set "remove" to True

    Class Figure attributes :-> a, b, step , remove
    ** 
    1) a -> start index name
    2) b -> end index name
    3) step -> iteration step 
    This three attributes give us the number of frames for example (0,101,1) will generate 100 pictures
    4) remove -> it deletes all pictures after program halts
    **
"""


class Figure:
    def __init__(self, a=1, b=101, step=1, remove=False):
        self.ls = {}
        figure = self.ChooseFigure()
        data = self.GetSides(figure)
        edge, colors= self.AskColor()
        self.fig = plt.figure()
        if figure == 1:
            for i in np.arange(a, b, step):
                fill = colors[i% len(colors)]
                self.DrawTriangle(data, edge, colors,i)
                print(str(i) + "%...")

            self.SaveTriangleGif(a, b, step)
            self.SaveTriangleVideo(a, b, step)
            if remove:
                self.DeleteTrianglePicture(a, b, step)

        elif figure == 2:
            for i in np.arange(a, b, step):
                fill = colors[i% len(colors)]
                self.DrawRectangle(data, edge, colors, i)
                print(str(i) + "%...")
            self.SaveRectangleGif(a, b, step)
            self.SaveRectangleVideo(a, b, step)
            if remove:
                self.DeleteRectanglePicture(a, b, step)

        elif figure == 3:
            pass

        else:
            for i in np.arange(a, b, step):
                fill = colors[i % len(colors)]
                self.DrawCircle(data, edge, colors,i)
                print(str(i) + "%...")
            self.SaveCircleGif(a, b, step)
            self.SaveCircleVideo(a, b, step)
            if remove:
                self.DeleteCirclePicture(a, b, step)
            pass

    def ColorFader(self, c1, c2, mix=0):
        c1 = np.array(matplotlib.colors.to_rgb(c1))
        c2 = np.array(matplotlib.colors.to_rgb(c2))
        return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)

    def DeleteCirclePicture(self, a, b, step):
        directory = 'Circle'
        for i in np.arange(a, b, step):
            f = os.path.join(directory, "circle" + str(i) + ".png")
            if os.path.exists(f):
                os.remove(f)
        return True

    def DeleteTrianglePicture(self, a, b, step):
        directory = 'Triangle'
        for i in np.arange(a, b, step):
            f = os.path.join(directory, "triangle" + str(i) + ".png")
            if os.path.exists(f):
                os.remove(f)
        return True

    def DeleteRectanglePicture(self, a, b, step):
        directory = 'Rectangle'
        for i in np.arange(a, b, step):
            f = os.path.join(directory, "rectangle" + str(i) + ".png")
            if os.path.exists(f):
                os.remove(f)
        return True

    def SaveTriangleVideo(self, a, b, step):
        directory = 'Triangle'
        with imageio.get_writer('triangle.mp4', fps=20) as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "triangle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def SaveTriangleGif(self, a, b, step):
        directory = 'Triangle'
        with imageio.get_writer('triangle.gif', mode='I') as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "triangle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def SaveRectangleGif(self, a, b, step):
        directory = 'Rectangle'
        with imageio.get_writer('rectangle.gif', mode='I') as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "rectangle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def SaveRectangleVideo(self, a, b, step):
        directory = 'Rectangle'
        with imageio.get_writer('rectangle.mp4', fps=20) as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "rectangle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def SaveCircleGif(self, a, b, step):
        directory = 'Circle'
        with imageio.get_writer('circle.gif', mode='I') as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "circle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def SaveCircleVideo(self, a, b, step):
        directory = 'Circle'
        with imageio.get_writer('circle.mp4', fps=20) as writer:
            for i in np.arange(a, b, step):
                f = os.path.join(directory, "circle" + str(i) + ".png")
                image = imageio.imread(f)
                writer.append_data(image)

    def AskColor(self):
        print()
        edge = input("Now Select Color for edges: ")
        print()
        fill = input("Now Select Colors to fill: ").split(" ")
        for i in range(len(fill)):
            if not is_color_like(fill[i]):
                fill[i] = "blue"
        if not is_color_like(edge):
            edge = "black"
        return edge, fill

    def DrawTriangle(self, data, edge, fill, percentage):
        if percentage > 100:
            percentage = 100

        domainForB = np.arange(0, data["B"][0] + 0.001, 0.001)
        domainForC = np.arange(0, data["C"][0] + 0.001, 0.001)
        domainForA = np.arange(data["B"][0], data["C"][0] + 0.001, 0.001)

        y1 = self.LeftTriangle(domainForB, data)
        if len(y1) == 0:
            plt.plot([0, 0], [0, data["B"][1]], color=edge)
        else:
            plt.plot(domainForB, y1, color=edge)
        y2 = self.RightTriangle(domainForA, data)

        height = data["B"][1] * percentage / 100
        x_left = self.GetLeftX(height, data)
        x_right = self.GetRightX(height, data)
        newdom = np.arange(0,x_left+0.001,0.001)
        newdom2 = np.arange(x_right, data["C"][0] + 0.001, 0.001)
        newY = self.LeftTriangle(newdom,data)
        newY2 = self.RightTriangle(newdom2,data)
        #plt.plot([x_left, x_right], [height, height], color=fill)

        plt.plot(domainForC, [0 for i in range(len(domainForC))], color=edge)
        plt.plot(domainForA, y2, color=edge)
        plt.xlim(-0.5, data["C"][0] + 0.5)
        plt.ylim(-0.5, data["B"][1] + 0.5)

        i = 0
        ground = 0
        h2 = 0
        while h2 <= height:
            h2 = data["B"][1] * i / 100
            #plt.fill_between(domain, ground, h2, color=fill[i % len(fill)])

            plt.fill_between(newdom, ground, newY, where=h2<=newY,color=fill[i % len(fill)])

            plt.fill_between(newdom2, ground, newY2, where=h2<=newY2,color=fill[i % len(fill)])
            plt.fill_between([x_left, x_right], ground, h2, color=fill[i % len(fill)])

            ground = h2
            i += 1

        plt.axis("off")
        plt.savefig("Triangle/triangle" + str(percentage) + ".png")
        # plt.show()


    def DrawRectangle(self, data, edge, fill, percentage):
        if percentage > 100:
            percentage = 100

        domain = [0, data["C"][0]]
        height = data["C"][1] * percentage / 100
        plt.plot([0, 0], [0, data["C"][1]], color=edge)
        plt.plot([data["C"][0], data["C"][0]], [0, data["C"][1]], color=edge)
        plt.plot([0, data["C"][0]], [0, 0], color=edge)
        plt.plot([0, data["C"][0]], [data["C"][1], data["C"][1]], color=edge)
        # plt.plot(domain, [height,height], color=fill)
        plt.xlim(-0.5, data["C"][0] + 0.5)
        plt.ylim(-0.5, data["C"][1] + 0.5)
        # plt.fill_between(domain, 0, height, color=fill)

        i = 0
        ground = 0
        h2 = 0
        while h2<= height:
            h2 = data["C"][1] * i / 100
            plt.fill_between(domain,ground,h2, color=fill[i % len(fill)])
            ground = h2
            i += 1


        plt.axis("off")
        plt.savefig("Rectangle/rectangle" + str(percentage) + ".png")


    def DrawCircle(self, data, edge, fill, percentage):
        if percentage > 100:
            percentage = 100
        percentage = 100 - percentage
        radius = data["radius"]
        domain = np.arange(-radius, radius + 0.01, 0.01)
        y1 = np.sqrt(np.power(radius, 2) - np.power(domain, 2))
        y2 = -np.sqrt(np.power(radius, 2) - np.power(domain, 2))
        if not data["full"]:
            plt.plot(domain, y2, color=edge)
            plt.plot(domain, [0 for i in domain], color=edge)
            height = -radius * percentage / 100
            newX = np.sqrt(np.power(radius, 2) - np.power(height, 2))
            newDomain = np.arange(-newX, newX + 0.01, 0.01)
            newY = -np.sqrt(np.power(radius, 2) - np.power(newDomain, 2))


            # plt.fill_between(newDomain, newY, height, color=fill[0])
            i = 0
            ground = -radius
            h2 = 0
            while h2 >= height:
                h2 =-radius * i / 100
                print(h2)
                plt.fill_between(domain, ground, height, where=ground >= newY, color=fill[i % len(fill)])
                ground = h2
                i += 1

        else:  # Chasasworebelia ....
            plt.plot(domain, y2, color=edge)
            plt.plot(domain, y1, color=edge)
            height = -radius * percentage / 100
            newX = np.sqrt(np.power(radius, 2) - np.power(height, 2))
            newDomain = np.arange(-newX, newX + 0.01, 0.01)
            newY1 = -np.sqrt(np.power(radius, 2) - np.power(newDomain, 2))

            plt.fill_between(newDomain, newY1, height)
            plt.fill_between(newDomain, newY1, height)

        plt.xlim(-0.5 - radius, radius + 0.5)
        plt.ylim(-0.5 - radius, radius + 0.5)

        plt.axis("off")
        # plt.savefig("Circle/circle" + str(percentage) + ".png")
        plt.show()

    def GetLeftX(self, y, data):
        if (data["B"][0] != 0):
            k = data["B"][1] / data["B"][0]
        else:
            return 0
        return y / k

    def GetRightX(self, y, data):
        k = data["B"][1] / (data["B"][0] - data["C"][0])
        b = -data["C"][0] * k
        return (y - b) / k

    def ChooseFigure(self):
        print("""Hello At First try to choose which figure you want to visualise
            1) Triangle
            2) Rectangle
            3) Polygon
            4) Circle
        """)
        figure = input("Please Enter The number: ")
        if (figure not in ["1", "2", "3", "4"]):
            return self.ChooseFigure()
        return int(figure)

    def GetSides(self, figure):
        if figure == 1:  # Triangle
            print("""Now Enter Side lengths of your triangle: """)
            A = input("First Side: ")
            B = input("Second Side: ")
            C = input("Third Side: ")
            if not (self.TriangleChecker(A, B, C)):
                return self.GetSides(figure)
            coord = self.GetCoordinateForTriangle(float(A), float(B), float(C))
            result = {
                "A": (0, 0),
                "B": coord,
                "C": (float(A), 0),
                "sides": [float(A), float(B), float(C)],
                "perimeter": float(A) + float(B) + float(C),
                "area": self.TriangleArea(float(A), float(B), float(C)),
            }
            return result
        elif figure == 2:  # Rectangle
            print("""Now Enter Side lengths of your rectangle: """)
            height = input("Height: ")
            width = input("Width: ")
            if not self.RectangleChecker(height, width):
                return self.GetSides(figure)
            result = {
                "A": (0, 0),
                "B": (0, float(height)),
                "C": (float(width), float(height)),
                "D": (float(width), 0),
                "sides": [float(height), float(width)],
                "perimeter": 2 * float(height) + 2 * float(width),
                "area": float(height) * float(width)
            }
            return result
        else:
            print("""Now Enter radius of your circle: """)
            radius = input("radius: ")
            if not radius.isnumeric():
                return self.GetSides(figure)
            print()
            print("Choose if you want a full circle ")
            style = input("Y/N: ")
            if style.lower() != "y":
                style = False
            else:
                style = True
            result = {
                "radius": float(radius),
                "area": np.power(float(radius), 2) * np.pi,
                "length": 2 * np.pi * float(radius),
                "full": style
            }
            return result

    def RectangleChecker(self, height, width):
        if not (height.isnumeric() or width.isnumeric()):
            return False
        if float(height) <= 0 or float(width) <= 0:
            return False
        return True

    def TriangleChecker(self, a, b, c):
        if not (a.isnumeric() or b.isnumeric() or c.isnumeric()): return False
        if (a == "" or b == "" or c == ""): return False
        smallest, medium, biggest = sorted([float(a), float(b), float(c)])
        return smallest + medium >= biggest and all(s > 0 for s in [float(a), float(b), float(c)])

    def GetCoordinateForTriangle(self, a, b, c):
        x = (np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a)
        y = np.sqrt(np.power(b, 2) - np.power(x, 2))
        return (x, y)

    def TriangleArea(self, a, b, c):
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def LeftTriangle(self, x, data):
        if (data["B"][0] != 0):
            k = data["B"][1] / data["B"][0]
        else:
            return []
        return np.array(x) * k

    def RightTriangle(self, x, data):
        k = data["B"][1] / (data["B"][0] - data["C"][0])
        b = -data["C"][0] * k
        return np.array(x) * k + b

Figure()
