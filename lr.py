"""
Script for building theano graph for Logistic Regression
Carmine Paolino
Aug, 2014
"""
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class LR():

    def __init__(self, rand, n_inputs=513, n_outputs=10, activation='sigmoid', output_layer='sigmoid'):
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.sizes = [self.n_inputs] + [self.n_outputs]
        self.numpy_rng = rand
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 10))
        self.output_layer = output_layer
        self.initialize_params()
        self.set_activation(activation)
        self.build_graph()

    def initialize_params(self,):
        self.W = []
        self.b = []
        for i in xrange(len(self.sizes) - 1):
            input_size = self.sizes[i]
            output_size = self.sizes[i + 1]
            W_init = numpy.asarray(self.numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (input_size + output_size)),
                                                          high=4 *
                                                          numpy.sqrt(
                                                              6. / (input_size + output_size)),
                                                          size=(input_size, output_size)),
                                   dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W_%d' % (i), borrow=True)
            b = theano.shared(value=numpy.zeros(output_size, dtype=theano.config.floatX), name='b_%d' % (i),
                              borrow=True)
            self.W.append(W)
            self.b.append(b)
        self.params = self.W + self.b

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = lambda x: T.nnet.sigmoid(x)
        elif activation == 'ReLU':
            self.activation = lambda x: T.maximum(0., x)
        else:
            print 'Activation must be sigmoid or ReLU. Quitting'
            sys.exit()

    def fprop(self, inputs, output_layer='sigmoid'):
        h = []
        h.append(self.activation(T.dot(inputs, self.W[0]) + self.b[0]))
        if output_layer == 'sigmoid':
            h.append(T.nnet.sigmoid(T.dot(h[-1], self.W[-1]) + self.b[-1]))
        elif output_layer == 'softmax':
            h.append(T.nnet.softmax(T.dot(h[-1], self.W[-1]) + self.b[-1]))
        else:
            print 'Output layer must be either sigmoid or softmax. Quitting.'
            sys.exit()
        return h[-1]

    def build_graph(self,):
        self.z = self.fprop(self.x, output_layer='sigmoid')
        #L = -T.sum(self.z*T.log(self.y) + (1-self.z)*T.log(1-self.y),axis=1)
        L = -T.sum(self.y * T.log(self.z) + (1 - self.y)
                   * T.log(1 - self.z), axis=1)
        #L = T.sum((self.z-self.y)**2,axis=1)
        self.cost = T.mean(L)
        self.preds = T.max(self.z, axis=1)
        self.acc = T.neq(
            T.argmax(self.z, axis=1), T.argmax(self.y, axis=1)).mean()
        # return [self.x,self.y],self.cost,self.params

if __name__ == '__main__':
    test = LR()
    _, _, _ = test.build_graph()
