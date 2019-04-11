import os
import unittest
from scripts import train

cuda_on = 1

class TestsOmniglot(unittest.TestCase):

    def test_1_shot_5_way_with_restore(self):
        config = {
            'data.dataset': 'omniglot',
            'data.split': 'vinyals',
            'data.train_way': 5,
            'data.batch': 32,
            'data.train_support': 5,
            'data.train_query': 1,
            'data.test_way': 5,
            'data.test_support': 5,
            'data.test_query': 1,
            'data.episodes': 1,
            'data.cuda': cuda_on,
            'data.gpu': 0,
            'model.x_dim': '28,28,1',
            'model.lstm_size': 32,
            'model.save_dir': './omniglot_test',
            'train.epochs': 1,
            'train.optim_method': 'Adam',
            'train.lr': 0.001,
            'train.patience': 100,
            'train.restore': 0,
            'train.log_dir': './logs'
        }
        train(config)

        # Train after restoring
        # TODO: It is probably works only on GPU, should fix it later
        config['train.restore'] = 1
        train(config)

    def test_5_shot_5_way(self):
        config = {
            'data.dataset': 'omniglot',
            'data.split': 'vinyals',
            'data.train_way': 5,
            'data.batch': 32,
            'data.train_support': 5,
            'data.train_query': 5,
            'data.test_way': 5,
            'data.test_support': 5,
            'data.test_query': 5,
            'data.episodes': 1,
            'data.cuda': cuda_on,
            'data.gpu': 0,
            'model.x_dim': '28,28,1',
            'model.lstm_size': 32,
            'model.save_dir': './omniglot_test',
            'train.epochs': 1,
            'train.optim_method': 'Adam',
            'train.lr': 0.001,
            'train.patience': 100,
            'train.restore': 0,
            'train.log_dir': './logs'
        }
        train(config)