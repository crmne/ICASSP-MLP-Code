"""
Parameters
"""
import os
import shutil


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
    state['learning_rate'] = 0.01
    state['num_epochs'] = 200  # number of epochs for SGD
    state['save'] = True  # bool, saves the best model and the costs and a bunch of other stuff if True
    state['output_folder'] = "4"  # String or None, where to save those infos
    state['lr_update'] = False  # bool, updates learning rate if True
    state['batch_size'] = None  # minibatch size used for SGD
    state['mom_rate'] = 0.2  # momentum rate
    state['plot'] = True  # bool, plots the costs in real time if True
    state['folds'] = 10  # number of folds to use
    state['songs_per_genre'] = 1  # number of songs per genre to be selected, or None to select all
    state['seed'] = 1  # number or None, set a global seed to make experiments reproducible
    state['train_valid_ratio'] = 0.8  # ratio between training and validation splits

    if state['output_folder']:
        state['output_folder'] = os.path.join('experiments', state['output_folder'])
    return state


def get_state_string():
    from collections import OrderedDict
    state = OrderedDict(sorted(get_state().items()))
    return "\n".join(["%s = %s" % (k, v) for k, v in state.iteritems()])


def save_state():
    output_folder = get_state()['output_folder']
    if output_folder and output_folder != "":
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_path = os.path.join(output_folder, 'state.py')
        shutil.copy(os.path.realpath(__file__), save_path)


if __name__ == '__main__':
    print get_state_string()
    save_state()
