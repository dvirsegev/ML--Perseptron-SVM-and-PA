import sys
from random import shuffle
import numpy as np


class Algorithm(object):
    "Function that swtich gender to a number"

    def switch(x, listresult):
        if (x == 'M'):
            return listresult[x]
        if (x == 'F'):
            return listresult[x]
        if (x == 'I'):
            return listresult[x]

    "Function that return a dictionary that the key is gender and the value is the prob of the certain gender"

    def checkRow(tmp):
        generalcount = 0
        Mcount = 0
        Fcount = 0
        Icount = 0
        for input in range(tmp.shape[0]):
            input = tmp[input][0]
            if (input == 'M'):
                Mcount += 1
            if (input == 'F'):
                Fcount += 1
            if (input == 'I'):
                Icount += 1
            generalcount += 1
        listresult = {
            "M": Mcount / generalcount,
            "F": Fcount / generalcount,
            "I": Icount / generalcount,
        }
        return listresult

    " Function of the test about the test examples"

    def test(self, inputext, w):
        x = []
        tmp = np.loadtxt(inputext, dtype='str', delimiter=',')
        listresult = Algorithm.checkRow(tmp)
        # switch the letter of the gender.
        for array in range(tmp.shape[0]):
            tmp[array][0] = Algorithm.switch(tmp[array][0], listresult)
        inputarray = tmp.astype(float)
        z = np.std(inputarray, axis=0)
        for i in range(z.size):
            if (z[i] is 0):
                z[i] = 1
        inputarray = (inputarray - np.mean(inputarray, axis=0)) / np.std(
            inputarray, axis=0)
        for inputs in inputarray:
            y_hat = np.argmax(np.dot(w, inputs))
            x.append(y_hat)
        return x


class Perceptron(Algorithm):
    "initialize the examples."

    def setup(self, train_x, train_y):
        tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
        listresult = Algorithm.checkRow(tmp)
        # switch the letter of the gender.
        for x in range(tmp.shape[0]):
            tmp[x][0] = Algorithm.switch(tmp[x][0], listresult)
        self.training_inputs = tmp.astype(float)
        self.training_inputs = (self.training_inputs - np.mean(self.training_inputs, axis=0)) / np.std(
            self.training_inputs, axis=0)
        self.labels = np.loadtxt(train_y, dtype=int)
        # shuffle examples
        c = list(zip(self.training_inputs, self.labels))
        shuffle(c)
        self.training_inputs, self.labels = zip(*c)
        # set width
        self.w = np.zeros((3, 8), dtype=float)
        self.learning_rate = 0.1

    "train the examples by perceptron algorithm."
    def train(self):
        for set in range(100):
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.w[label, :] += self.learning_rate * inputs
                    self.w[y_hat, :] -= self.learning_rate * inputs

    def weight(self):
        return self.w


class SVM(Algorithm):
    "initialize the examples."

    def setup(self, train_x, train_y):
        tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
        listresult = Algorithm.checkRow(tmp)
        # switch the letter of the gender.
        for x in range(tmp.shape[0]):
            tmp[x][0] = Algorithm.switch(tmp[x][0], listresult)
        self.training_inputs = tmp.astype(float)
        self.training_inputs = (self.training_inputs - np.mean(self.training_inputs, axis=0)) / np.std(
            self.training_inputs, axis=0)
        self.labels = np.loadtxt(train_y, dtype=int)
        # set width
        self.w = np.zeros((3, 8), dtype=float)
        self.learning_rate = 0.1
        self.lamda = 2

    "train the examples by perceptron algorithm."

    def train(self):
        for set in range(50):
            c = list(zip(self.training_inputs, self.labels))
            shuffle(c)
            self.training_inputs, self.labels = zip(*c)
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.w[label, :] = (1 - self.lamda * self.learning_rate) * self.w[label,
                                                                               :] + self.learning_rate * inputs
                    self.w[y_hat, :] = (1 - self.lamda * self.learning_rate) * self.w[label,
                                                                               :] - self.learning_rate * inputs

    def weight(self):
        return self.w


class PA(Algorithm):
    "initialize the examples."

    def setup(self, train_x, train_y):
        tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
        listresult = Algorithm.checkRow(tmp)
        # switch the letter of the gender.
        for x in range(tmp.shape[0]):
            tmp[x][0] = Algorithm.switch(tmp[x][0], listresult)
        self.training_inputs = tmp.astype(float)
        if (np.std(self.training_inputs) != 0):
            self.training_inputs = (self.training_inputs - np.mean(self.training_inputs, axis=0)) / (2 * np.std(
                self.training_inputs, axis=0))
        self.labels = np.loadtxt(train_y, dtype=int)
        # set width
        self.w = np.zeros((3, 8), dtype=float)
        self.lamda = 0

    "train the examples by perceptron algorithm."

    def train(self):
        for set in range(50):
            c = list(zip(self.training_inputs, self.labels))
            shuffle(c)
            self.training_inputs, self.labels = zip(*c)
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.lamda = max(0, 1 - np.dot(self.w[label], inputs) + np.dot(self.w[y_hat], inputs)) \
                                 / (np.linalg.norm(inputs) ** 2)
                    self.w[label, :] += self.lamda * inputs
                    self.w[y_hat, :] -= self.lamda * inputs

    def weight(self):
        return self.w


def main():
    inputx = sys.argv[1]
    inputy = sys.argv[2]
    inputext = sys.argv[3]
    f = open("result.txt", "a")
    "set up the Algorithm of perceptron and svm and PA"
    perceptron = Perceptron()
    perceptron.setup(inputx, inputy)
    perceptron.train()
    w = perceptron.weight()
    per_values = perceptron.test(inputext, w)
    svm = SVM()
    svm.setup(inputx, inputy)
    svm.train()
    w = svm.weight()
    svm_values = svm.test(inputext, w)
    pa = PA()
    pa.setup(inputx, inputy)
    pa.train()
    w = pa.weight()
    pa_values = pa.test(inputext, w)
    f.close()
    for per, svm, pa in zip(per_values, svm_values, pa_values):
        print("perceptron: " + str(per) + ", svm: " + str(svm) + ", pa: " + str(pa))


if __name__ == "__main__":
    main()
