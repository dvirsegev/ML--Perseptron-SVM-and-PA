from random import shuffle

import numpy as np


def switch(x):
    if (x=='M'):
        return 0
    if (x == 'F'):
        return 1
    if (x == 'I'):
        return 2


def printtothefile(file,text):
    f = open(file, "a")
    f.write(text)
    f.close()



class Perceptron(object):

    #initilizae the examples.
    def setup(self, train_x,train_y):
        tmp = np.loadtxt(train_x,dtype='str',delimiter=',')
        # switch the letter of the gender.
        for x in range(tmp.shape[0]):
            tmp[x][0]=switch(tmp[x][0])
        self.training_inputs= tmp.astype(float)
        self.labels = np.loadtxt(train_y,dtype=int)
        self.training_inputs = (self.training_inputs - np.mean(self.training_inputs,axis=0))/ np.std(
            self.training_inputs,axis=0)
        # shuffle examples
        x=np.max(self.labels[0])
        #set width
        self.w = np.zeros ( ( 3, 8) , dtype=float)
        self.learning_rate = 0.1

    def train(self):
        s=""
        count=0
        for set in range(50):
            c = list(zip(self.training_inputs, self.labels))
            shuffle(c)
            self.training_inputs, self.labels = zip(*c)
            for inputs, label in zip(self.training_inputs, self.labels):
                y_hat = np.argmax(np.dot(self.w, inputs))
                if (y_hat != label):
                    self.w[label, :] += self.learning_rate * inputs
                    self.w[y_hat, :] -= self.learning_rate * inputs
                #printtothefile("result.txt","perceptron: " + str(y_hat) + "\n")
                s+= str(y_hat)+"\n"
        for inputs, label in zip(self.training_inputs, self.labels):
            y_hat = np.argmax(np.dot(self.w, inputs))
            if (y_hat != label):
                count+=1
        avg= count/3286;
        print(avg)


        

def main():
    perceptron = Perceptron()
    perceptron.setup("train_x.txt","train_y.txt")
    perceptron.train()

if __name__ == "__main__":
    main()
