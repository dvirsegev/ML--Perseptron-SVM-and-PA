import sys
from random import shuffle
import numpy as np


class Algorithm(object):

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def switch(x,listresult):
        if (x == 'M'):
            return listresult[x]
        if (x == 'F'):
            return listresult[x]
        if (x == 'I'):
            return listresult[x]

    def checkRow(tmp):
        generalcount=0
        Mcount=0
        Fcount=0
        Icount=0
        for input in range (tmp.shape[0]):
            input = tmp[input][0]
            if (input == 'M'):
                Mcount+=1
            if (input == 'F'):
                Fcount+=1
            if (input == 'I'):
                Icount += 1
            generalcount+=1
        listresult={
            "M": Mcount/generalcount,
             "F": Fcount/generalcount,
            "I": Icount / generalcount,
        }
        return listresult
    def printtothefile(file, text):
        file.write(text)
        file.write("\n")

    def test(self, inputext,w):
            x = []
            tmp = np.loadtxt(inputext, dtype='str', delimiter=',')
            listresult = Algorithm.checkRow(tmp)
            # switch the letter of the gender.
            for array in range(tmp.shape[0]):
                tmp[array][0] = Algorithm.switch(tmp[array][0], listresult)
            inputarray = tmp.astype(float)
            z=np.std(inputarray, axis=0)
            for i in range (z.size):
                if(z[i] is 0):
                    z[i]=1
            inputarray = (inputarray - np.mean(inputarray, axis=0)) / np.std(
                inputarray, axis=0)
            for inputs in inputarray:
                y_hat = np.argmax(np.dot(w, inputs))
                x.append(y_hat)
            return x


class Perceptron(Algorithm):

    # initilizae the examples.
    def setup(self, train_x, train_y):
            tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
            listresult = Algorithm.checkRow(tmp)
            # switch the letter of the gender.
            for x in range(tmp.shape[0]):
                tmp[x][0] = Algorithm.switch(tmp[x][0],listresult)
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
    def train(self):
        count=0
        for set in range(100):
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.w[label, :] += self.learning_rate * inputs
                    self.w[y_hat, :] -= self.learning_rate * inputs

    def weight(self):
        return self.w


class SVM(Algorithm):
    # initilizae the examples.
    def setup(self, train_x, train_y):
            tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
            listresult = Algorithm.checkRow(tmp)
            # switch the letter of the gender.
            for x in range(tmp.shape[0]):
                tmp[x][0] = Algorithm.switch(tmp[x][0],listresult)
            self.training_inputs = tmp.astype(float)
            self.training_inputs = (self.training_inputs - np.mean(self.training_inputs, axis=0)) / np.std(
            self.training_inputs, axis=0)
            self.labels = np.loadtxt(train_y, dtype=int)
            # set width
            self.w = np.zeros((3, 8), dtype=float)
            self.learning_rate = 0.1
            self.lamda=2

    def train(self):
       # s = ""
        count = 0
        for set in range(50):
            c = list(zip(self.training_inputs, self.labels))
            shuffle(c)
            self.training_inputs, self.labels = zip(*c)
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.w[label, :] =(1-self.lamda * self.learning_rate) *self.w[label,:] + self.learning_rate * inputs
                    self.w[y_hat, :] =(1-self.lamda * self.learning_rate) *self.w[label,:] -self.learning_rate * inputs

    def weight(self):
        return self.w

class PA(Algorithm):
    # initilizae the examples.
    def setup(self, train_x, train_y):
        try:
            tmp = np.loadtxt(train_x, dtype='str', delimiter=',')
            listresult = Algorithm.checkRow(tmp)
            # switch the letter of the gender.
            for x in range(tmp.shape[0]):
                tmp[x][0] = Algorithm.switch(tmp[x][0],listresult)
            self.training_inputs = tmp.astype(float)
            if(np.std(self.training_inputs) != 0):
                self.training_inputs = (self.training_inputs - np.mean(self.training_inputs, axis=0)) / (2 * np.std(
                self.training_inputs, axis=0) )
            self.labels = np.loadtxt(train_y, dtype=int)
            # set width
            self.w = np.zeros((3, 8), dtype=float)
            self.lamda=0
        except Exception as e:
            print(type(e))

    def train(self):
        s = ""
        count = 0
        for set in range(50):
            c = list(zip(self.training_inputs, self.labels))
            shuffle(c)
            self.training_inputs, self.labels = zip(*c)
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.lamda= max(0,1-np.dot( self.w[label],inputs) + np.dot( self.w[y_hat],inputs))\
                                /(np.linalg.norm(inputs)**2)
                    self.w[label, :] +=self.lamda *inputs
                    self.w[y_hat, :] -=self.lamda *inputs
                # printtothefile("result.txt","perceptron: " + str(y_hat) + "\n")
                s += str(y_hat) + "\n"

    def weight(self):
        return self.w


def main():
        inputx=sys.argv[1]
        inputy=sys.argv[2]
        inputext=sys.argv[3]
        count = Algorithm.file_len(inputx)
        f = open("result.txt", "a")
        # perceptron execution
        perceptron = Perceptron()
        perceptron.setup(inputx, inputy)
        perceptron.train()
        w= perceptron.weight()
        per_values = perceptron.test(inputext,w)
        svm = SVM()
        svm.setup(inputx, inputy)
        svm.train()
        w = svm.weight()
        svm_values = svm.test(inputext,w)
        pa = PA()
        pa.setup(inputx, inputy)
        pa.train()
        w = pa.weight()
        pa_values = pa.test(inputext,w)
        for i in range (count):
            Algorithm.printtothefile(f,"perceptron: " + str(per_values[i]) + ", svm: " + str(svm_values[i]) +
                                     ", pa: " + str(pa_values[i]))
        f.close()
        f= open("result.txt",'r')
        content = f.readlines()
        for line in content:
            line=line.strip('\n')
            print(line)
if __name__ == "__main__":
    main()
