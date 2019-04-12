import os
import numpy as np
import pickle
from numpy.random import permutation


class MiniImagenetDataLoader(object):
    def __init__(self, data, batch, n_classes, n_way, n_support, n_query):
        self.data = data
        self.batch = batch
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        """
        Form episode data.

        Returns (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        """
        n_examples = 600
        x_support = np.zeros([self.batch, self.n_way * self.n_support,
                              84, 84, 3], dtype=np.float32)
        y_support = np.zeros([self.batch, self.n_way * self.n_support])
        x_query = np.zeros([self.batch, self.n_way * self.n_query,
                            84, 84, 3], dtype=np.float32)
        y_query = np.zeros([self.batch, self.n_way * self.n_query])

        for i_batch in range(self.batch):
            classes_ep = permutation(self.n_classes)[:self.n_way]
            x_support_batch = []
            y_support_batch = []
            x_query_batch = []
            y_query_batch = []
            for i, i_class in enumerate(classes_ep):
                selected = permutation(n_examples)[:self.n_support + self.n_query]
                x_support_batch.append(self.data[i_class, selected[:self.n_support]])
                y_support_batch += [i] * self.n_support
                x_query_batch.append(self.data[i_class, selected[self.n_support:]])
                y_query_batch += [i] * self.n_query
            x_support[i_batch, :, :, :, :] = np.vstack(x_support_batch)
            y_support[i_batch, :] = np.asarray(y_support_batch)
            x_query[i_batch, :, :, :, :] = np.vstack(x_query_batch)
            y_query[i_batch, :] = np.asarray(y_query_batch)

        return x_support, y_support, x_query, y_query


def load_mini_imagenet(data_dir, config, splits):
    """
    Load miniImagenet dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support examples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

        # Load images as numpy
        ds_filename = os.path.join(data_dir, 'data',
                                   f'mini-imagenet-cache-{split}.pkl')
        # load dict with 'class_dict' and 'image_data' keys
        with open(ds_filename, 'rb') as f:
            data_dict = pickle.load(f)

        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(data_dict['class_dict'])[0]
        data = np.zeros((len(data_dict['class_dict']),
                         len(data_dict['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(data_dict['class_dict'].items()):
            data[i, :, :, :, :] = data_dict['image_data'][v, :]

        # Normalization
        data /= 255.
        data[:, :, :, 0] = (data[:, :, :, 0] - 0.485) / 0.229
        data[:, :, :, 1] = (data[:, :, :, 1] - 0.456) / 0.224
        data[:, :, :, 2] = (data[:, :, :, 2] - 0.406) / 0.225

        batch = config['data.batch']
        data_loader = MiniImagenetDataLoader(data, batch, data.shape[0],
                                             n_way, n_support, n_query)
        ret[split] = data_loader

    return ret
