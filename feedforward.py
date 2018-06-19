import numpy as np
class Layer():

    # create an empty layer with the appropriate sizes    
    def __init__(self, size, size_next, alpha):
        self.weights = np.random.rand(size_next, size + 1)
        self.input = np.hstack((1,np.zeros(size))).reshape(-1,1)
        self.output = np.zeros(size_next)
        self.delta = np.zeros((size, 1))
        self.alpha = alpha
        print("creating layer with size", size, size_next)
        
    # forward pass
    def calculate(self, input_arr):
        self.input[1:] = input_arr.reshape(-1,1)
        self.output = self.sigmoid(np.dot(self.weights, self.input)).reshape(-1)
        return self.output

    # sigmoid
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-1 * t))
        
    # backward pass
    def backpropagate(self, d_next=[], y=[], final_layer=False):
        if(final_layer == True):
            self.delta = self.output * (1-self.output) * (y-self.output)
            self.delta = self.delta.reshape(-1,1)
            self.weights += self.alpha * np.dot(self.delta, self.input.T)
            # redefine delta to get backpropagation going
            self.delta = self.input * (1-self.input) *  \
                np.dot(self.weights.T, self.delta)
            self.delta = np.delete(self.delta, 0, 0)
        else:
            # get precalculated delta
            self.delta = d_next
            #self.delta = self.output * (1 - self.output) * np.dot(d_next, self.weights.T)
            self.weights += self.alpha * np.dot(self.delta, self.input.T)
            self.delta = self.input * (1-self.input) *  \
                np.dot(self.weights.T, self.delta.reshape(-1,1)).reshape(-1)
            # remove bias delta since we dont need to push it down
            self.delta = np.delete(self.delta, 0, 0)
        return self.delta

class NN():
    
    # create empty nn with all layers
    def __init__(self, size_input, size_hidden, size_output, alpha):
        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.layers = [Layer(size_input, size_hidden[0], alpha)]
        self.layers += [Layer(size_hidden[s], size_hidden[s + 1], alpha) \
                       for s in range(0,len(size_hidden)-1)]
        self.final_layer = Layer(size_hidden[-1], size_output, alpha)
    
    # run the training on one example
    def train(self,X,Y):
        for L in self.layers:
            X = L.calculate(X)

        pred = self.final_layer.calculate(X)
        d = self.final_layer.backpropagate(y=Y, final_layer=True)
        for i in range(len(self.layers)):
            d = self.layers[-i].backpropagate(d)
            
    # just output the forward pass
    def predict(self, X):
        
        for L in self.layers:
            X = L.calculate(X)

        return self.final_layer.calculate(X)

# this is the testing / hyperparameter tuning part

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import keras.utils

# load X
X_train = []
with open('train_x_small.csv','r') as f:
    for i,l in enumerate(f):
        X_train.append([int(float(a)) for a in l.split(',')])

X_train = np.array(X_train)

# load Y
Y_letter_true = []
with open('train_y.csv','r') as f:
    for l in f:
        Y_letter_true.append(l[:-1])

# create a one hot encoded vector for Y
num_classes = len(set(Y_letter_true))
mapping = {n:i for i,n in enumerate(list(set(Y_letter_true)))}
Y_train = keras.utils.to_categorical([mapping[n] for n in Y_letter_true], num_classes)

# run PCA (this could introduce a bias towards the dev set but we dont care since we have the test set)
pca = PCA(n_components=100)
X_train_trans = pca.fit_transform(X_train)

overall = []

# run an example evaluation for layer size
for i in np.linspace(10,500,10):
    print("Testing size", i)
    results = []
    # 5 times CV
    for c in range(5):
        # split in train and dev
        X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_train_trans, Y_train)
        X_train_train = X_train_train.astype('float32') / max([max(a) for a in X_train_train])
        X_train_test = X_train_test.astype('float32') / max([max(a) for a in X_train_train])
        print("Round", c)
        # create new nn with appropriate size
        n = NN(100,[int(i)],num_classes,0.55)
        # run training
        for x_t, y_t in zip(X_train_train, Y_train_train):
            n.train(x_t, y_t)
        print("Training completed")
        preds = []
        # evaluate
        for x in X_train_test:
            preds.append(n.predict(x))
        results.append([int(np.argmax(p) == np.argmax(Y_train_test)) for p in preds])
    overall.append(results)