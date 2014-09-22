"""
Trainer class
Siddharth Sigia
Feb,2014
C4DM
"""
import os
from preprocessing import PreProcessor
from mlp import MLP
from lr import LR
from sgd import SGD_Optimizer
from dataset import Dataset
from numpy.random import RandomState
import state
import argparse


class trainer():

    def __init__(self, state, rand):
        self.state = state
        self.dataset_dir = self.state.get('dataset_dir', '')
        self.list_dir = os.path.join(self.dataset_dir, 'lists')
        self.lists = {}
        self.lists['train'] = os.path.join(self.list_dir, 'train_1_of_1.txt')
        self.lists['valid'] = os.path.join(self.list_dir, 'valid_1_of_1.txt')
        self.lists['test'] = os.path.join(self.list_dir, 'test_1_of_1.txt')
        self.preprocessor = PreProcessor(self.dataset_dir)
        print 'Preparing train/valid/test splits'
        self.preprocessor.prepare_fold(
            self.lists['train'], self.lists['valid'], self.lists['test'])
        self.data = self.preprocessor.data
        self.targets = self.preprocessor.targets
        print 'Building model.'
        if self.state.get('model', 'MLP') == 'MLP':
            self.model = MLP(rand, n_inputs=self.state.get('n_inputs', 513),
                             n_outputs=self.state.get('n_ouputs', 10),
                             n_hidden=self.state.get('n_hidden', [50]),
                             activation=self.state.get(
                                 'activation', 'sigmoid'),
                             output_layer=self.state.get('sigmoid', 'sigmoid'),
                             dropout_rates=self.state.get('dropout_rates', None))
        elif self.state.get('model') == 'LR':
            self.model = LR(rand, n_inputs=self.state.get('n_inputs', 513),
                            n_outputs=self.state.get('n_ouputs', 10),
                            activation=self.state.get('activation', 'sigmoid'),
                            output_layer=self.state.get('sigmoid', 'sigmoid'))

    def train(self,):
        print 'Starting training.'
        print 'Initializing train dataset.'
        self.batch_size = self.state.get('batch_size', 20)
        train_set = Dataset(
            [self.data['train']],
            batch_size=self.batch_size,
            targets=[self.targets['train']])
        print 'Initializing valid dataset.'
        valid_set = Dataset(
            [self.data['valid']],
            batch_size=self.batch_size,
            targets=[self.targets['valid']])
        self.optimizer = SGD_Optimizer(
            self.model.params,
            [self.model.x, self.model.y],
            [self.model.cost, self.model.acc],
            momentum=self.state.get('momentum', False))
        lr = self.state.get('learning_rate', 0.1)
        num_epochs = self.state.get('num_epochs', 200)
        save = self.state.get('save', False)
        mom_rate = self.state.get('mom_rate', None)
        plot = self.state.get('plot', False)
        output_folder = self.state.get('output_folder', None)
        self.optimizer.train(train_set,
                             valid_set,
                             self.state.get('update_lr_params'),
                             lr,
                             num_epochs,
                             save,
                             output_folder,
                             self.state.get('lr_update'),
                             mom_rate,
                             plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains the neural network.")
    parser.add_argument("dataset_dir", help="/path/to/dataset_dir")
    args = parser.parse_args()

    state = state.get_state()
    state['dataset_dir'] = os.path.abspath(args.dataset_dir)

    rand = RandomState(state['seed'])
    print "Seed: %i" % rand.get_state()[1][0]  # ugly but works in numpy 1.8.1

    test = trainer(state, rand)
    test.train()
