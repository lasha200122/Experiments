import os
from matplotlib.colors import is_color_like
from win32api import GetSystemMetrics
import imageio.v2
import pygame

directory = "Game"
picture = "frame"
color = "red"
radius = 20
speed = 2
fps = 50
width = GetSystemMetrics(0) - 70
height = GetSystemMetrics(1) - 70
delete = False
backgroundPath = "frame0.png"
video_name = "video.mp4"


class Circle:
    def __init__(self, directory, picture, color, radius, speed, fps, width, height, delete, backgroundPath, video_name):

        # At first let's create folder to store our frames
        self.createFolder(directory)

        # Now let's choose format of the video
        format = self.choseFormat()

        # Count frames
        self.count = 0

        if format == 1:
            # Here we choose color of background
            bgColor = self.choseBGColor()

            # Get Staring Position for Circle
            position = self.chosePosition(radius, width, height)

            # Ask Direction
            direction = self.askDirection(position, radius, width, height)

            # Show Display
            self.showDisplay(width, height, fps, bgColor, position, radius, color, direction, speed, picture, directory)

            # Save Video
            self.saveVideo(video_name, fps, directory, picture)

            if delete:
                self.Delete(directory)

            return

        # Show Display 2
        self.showDisplay2(fps, radius, color, speed, picture, directory, backgroundPath)

        # Save Video
        self.saveVideo(video_name, fps, directory, picture)

        if delete:
            self.Delete(directory)

        return

    def askDirection(self, position, radius, width, height):

        if position == [radius, radius]:  #Left Upper Corner
            print()
            print("Now choose Direction")
            print(
                """
                1) Right
                2) Diagonal
                3) Down
                """)
            result = input("Write Number: ")
            if result in ["1","2","3"]:
                return (1, int(result))
            return self.askDirection(position, radius, width, height)

        elif position == [width - radius, radius]:  # Right Upper Corner
            print()
            print("Now choose Direction")
            print(
                """
                1) Left
                2) Diagonal
                3) Down
                """)
            result = input("Write Number: ")
            if result in ["1", "2", "3"]:
                return (2, int(result))
            return self.askDirection(position, radius, width, height)

        elif position == [radius, height - radius]:  # Left Down Corner
            print()
            print("Now choose Direction")
            print(
                """
                1) Up
                2) Diagonal
                3) Right
                """)
            result = input("Write Number: ")
            if result in ["1", "2", "3"]:
                return (3, int(result))
            return self.askDirection(position, radius, width, height)

        else:  # Right Down Corner
            print()
            print("Now choose Direction")
            print(
                """
                1) Left
                2) Diagonal
                3) Up
                """)
            result = input("Write Number: ")
            if result in ["1", "2", "3"]:
                return (4, int(result))
            return self.askDirection(position, radius, width, height)

    def createFolder(self, directory):
        print("------------------------------------------")
        print()
        print("Checking Folder...")
        if not os.path.exists(directory):
            print()
            print("Creating Folder...")
            os.makedirs(directory)
            print()
            print("Folder Created")
            return True
        print()
        print("Folder Already Exists")
        return False

    def choseFormat(self):
        print()
        print("Now choose: ")
        print("""1) Without Background Picture
2) With Background Picture""")
        result = input("Write 1 or 2: ")
        if result == "1" or result == "2":
            return int(result)
        return self.choseFormat()

    def choseBGColor(self):
        print()
        print("Choose color for background")
        result = input("Write Color: ")
        if is_color_like(result):
            return result
        return self.choseBGColor()

    def chosePosition(self, radius, width, height):
        print()
        print("Now choose starting position:")
        print(
            """
            1) Left Upper Corner
            2) Right Upper Corner
            3) Left Down Corner
            4) Right Down Corner
            """)
        result = input("Write Number: ")
        if result in ["1", "2", "3", "4"]:
            if result == "1":
                return [radius, radius]
            elif result == "2":
                return [width - radius, radius]
            elif result == "3":
                return [radius, height - radius]
            else:
                return [width - radius, height - radius]
        return self.chosePosition(radius, width, height)

    def showDisplay(self, width, height, fps, bgColor, position, radius, color, direction, speed, picture, directory):
        pygame.init()
        display = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        self.turnOn(fps, display, clock, bgColor, position, radius, color, direction, speed, width, height, picture, directory)
        pygame.quit()

    def showDisplay2(self, fps, radius, color, speed, picture, directory, backgroundPath):
        pygame.init()
        position = [163,66]
        bg = pygame.image.load(backgroundPath)
        display = pygame.display.set_mode((750, 425))
        clock = pygame.time.Clock()
        self.turnOn2(fps, display, clock, bg, radius, color, speed, picture, directory, position)
        pygame.quit()

    def turnOn(self, fps, display, clock, bgColor, position, radius, color, direction, speed, width, height, picture, directory):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            display.fill(bgColor)
            start = self.move(position, direction, speed, width, height, radius)
            pygame.draw.circle(display, color, position, radius, 0)
            pygame.display.update()
            clock.tick(fps)
            pygame.image.save(display, directory + "/" + picture + str(self.count) + ".png")
            self.count += 1
            if not start:
                return False

    def turnOn2(self, fps, display, clock, bg, radius, color, speed, picture, directory, position):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            display.blit(bg, (0, 0))
            self.move2(position, speed)
            radius += speed*0.1
            pygame.draw.circle(display, color, position, radius, 0)
            pygame.display.update()
            clock.tick(fps)
            pygame.image.save(display, directory + "/" + picture + str(self.count) + ".png")
            self.count += 1
            if position[0] >= 750:
                return

    def move(self, position, direction, speed, width, height, radius):
        if direction[0] == 1:
            if direction[1] == 1:
                position[0] += speed
                if position[0] >= width - radius:
                    return False
                return True
            elif direction[1] == 2:
                position[0] += speed
                position[1] = self.diagonal([radius, radius], [width - radius, height - radius], position[0])
                if position[0] >= width - radius:
                    return False
                return True
            else:
                position[1] += speed
                if position[1] >= height - radius:
                    return False
                return True
        elif direction[0] == 2:
            if direction[1] == 1:
                position[0] -= speed
                if position[0] <= radius:
                    return False
                return True
            elif direction[1] == 2:
                position[0] -= speed
                position[1] = self.diagonal([width - radius, radius], [radius, height - radius], position[0])
                if position[0] <= radius:
                    return False
                return True
            else:
                position[1] += speed
                if position[1] >= height + radius:
                    return False
                return True
        elif direction[0] == 3:
            if direction[1] == 1:
                position[1] -= speed
                if position[1] <= radius:
                    return False
                return True
            elif direction[1] == 2:
                position[0] += speed
                position[1] = self.diagonal([radius, height - -radius], [width - radius, radius], position[0])
                if position[0] >= width - radius:
                    return False
                return True
            else:
                position[0] += speed
                if position[0] >= width - radius:
                    return False
                return True
        else:
            if direction[1] == 1:
                position[0] -= speed
                if position[0] <= radius:
                    return False
                return True
            elif direction[1] == 2:
                position[0] -= speed
                position[1] = self.diagonal([radius, radius], [width - radius, height - radius], position[0])
                if position[0] <= radius:
                    return False
                return True
            else:
                position[1] -= speed
                if position[1] <= radius:
                    return False
                return True

    def move2(self, position, speed):
        position[0] += speed
        position[1] = self.diagonal([163,66], [739,423], position[0])

    def diagonal(self, point1, point2, x):
        k = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - k * point1[0]
        return k * x + b

    def saveVideo(self, video_name, fps, directory, picture):
        with imageio.get_writer(video_name, fps=fps) as writer:
            for i in range(self.count):
                f = os.path.join(directory, picture + str(i) + ".png")
                image = imageio.v2.imread(f)
                writer.append_data(image)

    def Delete(self, directory):
        if os.path.exists(directory):
            os.remove(directory)
            return True
        return False


Circle(directory, picture, color, radius, speed, fps, width, height, delete, backgroundPath, video_name)
