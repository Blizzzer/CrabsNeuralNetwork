import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.delta = 0.0


class Neuron:
    eta = 0.001
    alpha = 0

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def sigmoid(self, x):
        return 1/(1 + math.exp(-x * 1.0))

    def sigmoidDeriv(self, x):
            return x*(1.0 - x)

    def addError(self, err):
        self.error = self.error + err


    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForward(self):
        sumOut = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOut = sumOut + dendron.connectedNeuron.getOutput()*dendron.weight
        self.output = self.sigmoid(sumOut)

    def backPropagate(self):
        self.gradient = self.error * self.sigmoidDeriv(self.output)
        for dendron in self.dendrons:
            dendron.delta = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.delta
            dendron.weight = dendron.weight + dendron.delta
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0

class NeuralNetwork:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None)) #bias
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])


    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e **2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def feedForward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForward()

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getBinarizedResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output


def main():
    randomLearning = True
    expectedAverageError = 0.04
    numberOfIterations = 200000

    topology = []
    topology.append(5)
    topology.append(40)
    topology.append(2)

    Neuron.alpha = 0  # Momentum

    errorToPlot = []  # Plot with every error
    errorToPlotForRandom = []  # Plot with average error
    errorToPlotForRandom1 = []  # Plot with average error
    errorToPlotForRandom2 = []  # Plot with average error
    plt.figure(1)
    for iters in range(3):

        if iters == 0:
            Neuron.eta = 0.007  # Learning rate
        if iters == 1:
            Neuron.eta = 0.015  # Learning rate
            errorToPlotForRandom1 = errorToPlotForRandom
            errorToPlotForRandom = []
        elif iters == 2:
            Neuron.eta = 0.023  # Learning rate
            errorToPlotForRandom2 = errorToPlotForRandom
            errorToPlotForRandom = []
        net = NeuralNetwork(topology)


        #Data import
        inputs = np.genfromtxt("data3.csv", delimiter=';').tolist()
        # sex 0 = M , 1 = F
        s = np.array(
            [np.append(np.append(np.zeros(50), np.ones(50)), np.append(np.zeros(50), np.ones(50))).tolist()]).T.tolist()
        # type 0 = B , 1 = O
        t = np.array([np.append(np.zeros(100), np.ones(100)).tolist()]).T.tolist()

        outputs = np.append(t, s, 1).tolist()

        #Learning process
        if randomLearning:
            for z in range(numberOfIterations):
                rn = randint(0, len(inputs)-1)
                outputs = np.append(t, s, 1).tolist()
                net.setInput(inputs[rn])
                net.feedForward()
                net.backPropagate(outputs[rn])
                err = net.getError(outputs[rn])
                errorToPlot.append(err)
                if z%200 == 0:
                    err = 0
                    print("Iterations left: ", numberOfIterations-z)
                    for i in range(len(inputs)):
                        net.setInput(inputs[i])
                        net.feedForward()
                        err = err + net.getError(outputs[i])
                    avrerr = err/len(inputs)
                    errorToPlotForRandom.append(avrerr)
                    print("Average error: ", avrerr)
                    if avrerr < expectedAverageError:
                        break

        else:
            for z in range(int(numberOfIterations/200)):
                err = 0
                for i in range(len(inputs)):
                    net.setInput(inputs[i])
                    net.feedForward()
                    net.backPropagate(outputs[i])
                    err = err + net.getError(outputs[i])
                    avrerr = err/len(inputs)
                print("Error: ", err)
                print("Average error: ", avrerr)
                print("Iterations left: ", int(numberOfIterations/200)-z)
                errorToPlot.append(avrerr)

        #Testing process
        correctAnswers = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForward()
            print("Number of answer: ", i)
            print("Answer given by network: ", net.getBinarizedResults())
            print("Expected answer: ", outputs[i])
            print("----------")
            if net.getBinarizedResults() == outputs[i]:
                correctAnswers = correctAnswers + 1
        print("Number of correct answers: ", correctAnswers)

    plt.plot(range(1000), errorToPlotForRandom1, 'r', range(1000),
             errorToPlotForRandom2, 'b', range(1000), errorToPlotForRandom, 'g')

if __name__ == '__main__':
        main()