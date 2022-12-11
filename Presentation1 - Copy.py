import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy


class AnimRect(object):
    '''Animate a rectangle'''
    def __init__(self):
        self.fig = plt.figure(figsize = (5,5))
        # create the axes
        self.ax = plt.axes(xlim=(0,100), ylim=(0,100), aspect='equal')
        # create rectangle
        self.rect = plt.Rectangle((0,0), 5, 50,
                                   fill=True, color='gold', ec='blue')
        self.ax.add_patch(self.rect)
        self.y=[]
        for i in range(100):
            x = random.randint(0, 30)
            self.y.append(x)
        self.y = numpy.array(self.y)
        # print(self.y,len(self.y))
        self.call_animation()
        plt.show()

    # animation function
    def animate(self, i):
        self.rect = self.ax.fill_between(x = (0, 5), y1 = 0, y2 = self.y[i], color = 'red')
        return self.rect,

    def call_animation(self):
        # call the animator function
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=len(self.y), interval=0.1, blit=True, repeat=False)

def main():
    rect = AnimRect()

main()