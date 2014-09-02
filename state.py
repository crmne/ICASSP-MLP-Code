"""
Parameters
"""


def get_state():
    state = {}
    state['model'] = 'MLP'  # 'MLP' or 'LR'
    state['n_inputs'] = 513  # number of input neurons
    state['n_outputs'] = 10  # number of output neurons
    state['n_hidden'] = [50]  # list of numbers of neurons in hidden layers, e.g.: [50, 50, 50]
    state['activation'] = 'ReLU'  # 'sigmoid' or 'ReLU'
    state['output_layer'] = 'softmax'  # 'sigmoid' or 'softmax'
    state['dropout_rates'] = None  # list of dropout rates for each layer, None if no dropout is used
    state['momentum'] = False  # bool
    state['learning_rate'] = 0.1
    state['num_epochs'] = 500  # number of epochs for SGD
    state['save'] = True  # bool, saves the best model and the costs and a bunch of other stuff if True
    state['output_folder'] = None  # String or None, where to save those infos
    state['lr_update'] = True  # bool, updates learning rate if True
    state['batch_size'] = 20  # minibatch size used for SGD
    state['mom_rate'] = 0.9  # momentum rate
    state['plot'] = True  # bool, plots the costs in real time if True
    state['folds'] = 4  # number of folds to use
    state['songs_per_genre'] = 10  # number of songs per genre to be selected, or None to select all
    state['seed'] = 1234  # number or None, set a global seed to make experiments reproducible
    state['train_valid_ratio'] = 0.66  # ratio between training and validation splits
    return state


def print_state():
    from collections import OrderedDict
    state = OrderedDict(sorted(get_state().items()))
    for k, v in state.iteritems():
        print "%s = %s" % (k, v)


if __name__ == '__main__':
    print_state()
