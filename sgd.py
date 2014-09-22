"""
SGD optimizer class
Siddharth Sigtia
Feb,2014
C4DM
"""
import numpy
import theano
import theano.tensor as T
import cPickle
import os
from theano.compat.python2x import OrderedDict
import copy
import matplotlib.pyplot as plt
import state


class SGD_Optimizer():

    def __init__(self, params, inputs, costs, updates_old=None,
                 consider_constant=[], momentum=True):
        """
        params: parameters of the model
        inputs: list of symbolic inputs to the graph
        costs: list of costs to be evaluated. The first element MUST be the
         objective.
        updates_old: OrderedDict from previous graphs that need to be accounted
         for by SGD, typically when scan is used.
        consider_constant: list of theano variables that are passed on to the
         grad method. Typically RBM.
        """
        self.inputs = inputs
        self.params = params
        self.momentum = momentum
        if self.momentum:
            self.params_mom = []
            for param in self.params:
                param_init = theano.shared(
                    value=numpy.zeros(
                        param.get_value().shape,
                        dtype=theano.config.floatX),
                    name=param.name + '_mom')
                self.params_mom.append(param_init)
        self.costs = costs
        self.num_costs = len(costs)
        self.updates_old = updates_old
        self.consider_constant = consider_constant
        self.build_train_fn()

    def build_train_fn(self,):
        self.lr_theano = T.scalar('lr')
        self.grad_inputs = self.inputs + [self.lr_theano]
        if self.momentum:
            self.mom_theano = T.scalar('mom')
            self.grad_inputs = self.grad_inputs + [self.mom_theano]

        self.gparams = T.grad(
            self.costs[0],
            self.params,
            consider_constant=self.consider_constant)
        if not self.momentum:
            print 'Building SGD optimization graph without momentum'
            updates = OrderedDict((i, i - self.lr_theano * j)
                                  for i, j in zip(self.params, self.gparams))
        else:
            print 'Building SGD optimization graph with momentum'
            updates = OrderedDict()
            for param, param_mom, gparam in zip(self.params, self.params_mom, self.gparams):
                param_inc = self.mom_theano * \
                    param_mom - self.lr_theano * gparam
                updates[param_mom] = param_inc
                updates[param] = param + param_inc
        self.calc_cost = theano.function(self.inputs, self.costs)
        if self.updates_old:
            # To avoid updating the model dict if updates dict belongs to model
            # class, very unlikely case.
            self.updates_old = copy.copy(self.updates_old)
            self.updates_old.update(updates)
        else:
            self.updates_old = OrderedDict()
            self.updates_old.update(updates)

        self.f = theano.function(
            self.grad_inputs, self.costs, updates=self.updates_old)

    def train(self, train_set, valid_set, update_lr_params, learning_rate, num_epochs, save, output_folder, lr_update, mom_rate, plot, headless):
        self.best_cost = numpy.inf
        self.init_lr = learning_rate
        self.lr = numpy.array(learning_rate)
        self.mom_rate = mom_rate
        self.output_folder = output_folder
        self.train_set = train_set
        self.valid_set = valid_set
        self.save = save
        self.lr_update = lr_update
        update_type = update_lr_params['update_type']
        begin_anneal = update_lr_params['begin_anneal']
        min_lr = update_lr_params['min_lr']
        decay_factor = update_lr_params['decay_factor']
        # each element is an epoch, each element of an epoch is 1st train and
        # 2nd validation. of each train and validation there are 2 elements
        costs = []
        if plot:
            self.fig = plt.figure()
            if not headless:
                plt.ion()
                plt.show()
        try:
            for u in xrange(num_epochs):
                epoch = []

                epoch.append(self.perform_training())

                epoch.append(self.perform_validation())

                best_params = self.are_best_params(numpy.absolute(epoch[-1]))

                if lr_update:
                    self.update_lr(
                        u + 1, update_type=update_type, begin_anneal=begin_anneal, min_lr=min_lr, decay_factor=decay_factor)

                print "Learning rate = %f" % self.lr

                self.print_epoch(epoch, u, best_params)
                if plot:
                    self.update_plot(epoch, u, best_params)
                costs.append(epoch)

        except KeyboardInterrupt:
            print 'Training interrupted.'

        if self.save:
            self.save_costs(costs)
            if plot:
                self.save_costs_plot()

    def update_plot(self, epoch, epoch_n, best_cost=False):
        x = epoch_n + 1

        plt.subplot(2, 2, 1)
        plt.plot(x, epoch[0][0], 'r' + 'o' if best_cost else '.' + '-')
        plt.title('Training')
        plt.ylabel('Cost 0')

        plt.subplot(2, 2, 3)
        plt.plot(x, epoch[0][1], 'r' + 'o' if best_cost else '.' + '-')
        plt.ylabel('Cost 1')
        plt.xlabel('Epoch')

        plt.subplot(2, 2, 2)
        plt.title('Validation')
        plt.plot(x, epoch[1][0], 'r' + 'o' if best_cost else '.' + '-')

        plt.subplot(2, 2, 4)
        plt.plot(x, epoch[1][1], 'r' + 'o' if best_cost else '.' + '-')
        plt.xlabel('Epoch')

        plt.draw()

    def are_best_params(self, cost):
        # import pdb
        # pdb.set_trace()
        ret = (cost < self.best_cost).all()
        if ret:
            self.best_cost = cost
            if self.save:
                self.save_model()
        return ret

    def print_epoch(self, epoch, epoch_n, best_cost=False):
        print "== Epoch %i ==" % (epoch_n + 1)
        print "Training Results:"
        for i in xrange(len(epoch[0])):
            print "Cost %i: %f" % (i, epoch[0][i])
        print "Validation Results:"
        for i in xrange(len(epoch[1])):
            print "Cost %i: %f" % (i, epoch[1][i])
        if best_cost:
            print "Best Params!"

    def perform_training(self):
        cost = []
        for i in self.train_set.iterate(True):
            inputs = i + [self.lr]
            if self.momentum:
                inputs = inputs + [self.mom_rate]
            cost.append(self.f(*inputs))
        return numpy.mean(cost, axis=0)

    def perform_validation(self):
        cost = []
        for i in self.valid_set.iterate(True):
            cost.append(self.calc_cost(*i))
        return numpy.mean(cost, axis=0)

        # Using accuracy as metric
        this_cost = numpy.absolute(numpy.mean(cost, axis=0))[1]
        if this_cost < self.best_cost:
            self.best_cost = this_cost
            print 'Best Params!'
            if self.save:
                self.save_model()

    def save_model(self,):
        best_params = [param.get_value().copy() for param in self.params]
        if not self.output_folder:
            cPickle.dump(best_params, open('best_params.pickle', 'w'))
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder, 'best_params.pickle')
            cPickle.dump(best_params, open(save_path, 'w'))

    def save_costs(self, costs):
        if not self.output_folder:
            cPickle.dump(costs, open('costs.pickle', 'w'))
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder, 'costs.pickle')
            cPickle.dump(costs, open(save_path, 'w'))

    def save_costs_plot(self):
        if not self.output_folder:
            self.fig.savefig('costs.pdf')
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder, 'costs.pdf')
            self.fig.savefig(save_path, format='PDF')

    def update_lr(self, count, update_type, begin_anneal, min_lr, decay_factor):
        if update_type == 'annealed':
            scale_factor = float(begin_anneal) / count
            self.lr = self.init_lr * min(1., scale_factor)
        if update_type == 'exponential':
            new_lr = float(self.init_lr) / (decay_factor ** count)
            if new_lr < min_lr:
                self.lr = min_lr
            else:
                self.lr = new_lr
