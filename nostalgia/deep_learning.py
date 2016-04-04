import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self,sizes):
        self._num_layers = len(sizes)
        self._sizes = sizes
        self._biases = [np.random.randn(y, 1) for y in self._sizes[1:]]
        self._weights = [np.random.randn(y, x)/np.sqrt(x)
                         for x, y in zip(self._sizes[:-1], self._sizes[1:])]

    def feedforward(self, x):
        a = np.array(x).reshape(-1,1)
        for b, w in zip(self._biases, self._weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def backprop(self,x,y):
        x = np.array(x).reshape(-1,1)
        y = np.array(y).reshape(-1,1)
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (activations[-1]-y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self._weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

if __name__ == '__main__':
    s = [4,3,2]
    n = Network(s)
    x = [1,1,1,1]
    y = [2,2]

    nabla_b,nabla_w = n.backprop(x,y)

    # print n._biases[0]
    # print n._weights[0],x,sigmoid(np.dot(n._weights[0],x)) +  n._biases[0]