import numpy as np

class TestNetwork:
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights =  2* np.random.random((3,1)) - 1


    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return  x * (1-x)

    def train(self, training_inputs , training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustment = np.dot(training_inputs.T,error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def think(self,inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return  outputs


def game_mode(text):
    num = input(text)
    if not num.isdigit():
        return game_mode("please enter digit: ")
    if int(num) not in [1, 2]:
        return game_mode("please enter 1 or 2: ")
    return int(num)

def bound(text):
    num = input(text)
    if not num.isdigit():
        return bound("please enter digit: ")
    return int(num)

def main(game_on):
    if game_on == False:
        return "Done"
    mode = game_mode("Choose game mode \n User to guess the number choose 1: ")
    lower = bound("Choose lower bound: ")
    upper = bound("Choose upper bound: ")
    if mode == 1:
        return main(guess_user(lower, upper))
    return main(guess_computer(lower, upper))

def guess_user(lower, upper): # here goes your logic...
    return True

def guess_computer(lower, upper): # here goes your logic...
    return True
main(True)