"""
Parameters
"""
import os
import shutil
import argparse

# TODO: distribution of the weights


def get_state():
    state = {}
    state['model'] = 'MLP'  # 'MLP' or 'LR'
    state['n_inputs'] = 513  # number of input neurons
    state['n_outputs'] = 10  # number of output neurons
    # list of numbers of neurons in hidden layers, e.g.: [50, 50, 50]
    state['n_hidden'] = [50]
    state['activation'] = 'sigmoid'  # 'sigmoid' or 'ReLU'
    state['output_layer'] = 'sigmoid'  # 'sigmoid' or 'softmax'
    # list of dropout rates for each layer, None if no dropout is used
    state['dropout_rates'] = [0.5]
    state['momentum'] = True  # bool
    state['num_epochs'] = 5000  # number of epochs for SGD
    # bool, saves the best model and the costs and a bunch of other stuff if
    # True
    state['save'] = True
    # String or None, where to save those infos
    state['output_folder'] = "Sep22A"
    state['batch_size'] = 200  # minibatch size used for SGD
    state['mom_rate'] = 0.1  # momentum rate
    state['plot'] = True  # bool, plots the costs in real time if True
    state['folds'] = 10  # number of folds to use
    # number of songs per genre to be selected, or None to select all
    state['songs_per_genre'] = 10
    # number or None, set a global seed to make experiments reproducible
    state['seed'] = 1
    # ratio between training and validation splits
    state['train_valid_ratio'] = 0.8
    state['learning_rate'] = 0.002
    state['lr_update'] = True  # bool, updates learning rate if True
    state['update_lr_params'] = {
        'update_type': 'exponential',
        'begin_anneal': 500,
        'min_lr': 0.00001,
        'decay_factor': 1.0005
    }

    if state['output_folder']:
        state['output_folder'] = os.path.join(
            'experiments', state['output_folder'])
    return state


def get_ordered_state():
    from collections import OrderedDict
    return OrderedDict(sorted(get_state().items()))


def get_state_string():
    return "\n".join(["%s = %s" % (k, v) for k, v in get_ordered_state().iteritems()])


def escape_latex_string(latex_string):
    return str(latex_string).replace('_', '\\_')


def get_latex_string():
    return "\\\\\n".join(["%s & %s" % (escape_latex_string(k), escape_latex_string(v)) for k, v in get_ordered_state().iteritems()])


def save_state():
    output_folder = get_state()['output_folder']
    if output_folder and output_folder != "":
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_path = os.path.join(output_folder, 'state.py')
        shutil.copy(os.path.realpath(__file__), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Displays the state.")
    parser.add_argument(
        "format", nargs='?', choices=['text', 'latex'], default='text')
    args = parser.parse_args()

    if args.format == 'text':
        print get_state_string()
    elif args.format == 'latex':
        print get_latex_string()

    save_state()
