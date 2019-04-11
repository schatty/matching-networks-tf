import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image


class OmniglotDataLoader(object):
    def __init__(self, data, batch, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.batch = batch
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = 20
        x_support = np.zeros([self.batch, self.n_way * self.n_support, 28, 28, 1], dtype=np.float32)
        y_support = np.zeros([self.batch, self.n_way * self.n_support])
        x_query = np.zeros([self.batch, self.n_way * self.n_query, 28, 28, 1], dtype=np.float32)
        y_query = np.zeros([self.batch, self.n_way * self.n_query])

        for i_batch in range(self.batch):
            classes_ep = np.random.permutation(self.n_classes)[:self.n_way]
            x_support_batch = []
            y_support_batch = []
            x_query_batch = []
            y_query_batch = []
            for i, i_class in enumerate(classes_ep):
                selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
                x_support_batch.append(self.data[i_class, selected[:self.n_support]])
                y_support_batch += [i] * self.n_support
                x_query_batch.append(self.data[i_class, selected[self.n_support:]])
                y_query_batch += [i] * self.n_query
            x_support[i_batch, :, :, :, :] = np.vstack(x_support_batch)
            y_support[i_batch, :] = np.asarray(y_support_batch)
            x_query[i_batch, :, :, :, :] = np.vstack(x_query_batch)
            y_query[i_batch, :] = np.asarray(y_query_batch)

        return x_support, y_support, x_query, y_query


def class_names_to_paths(data_dir, class_names):
    """
    Return full paths to the directories containing classes of images.

    Args:
        data_dir (str): directory with dataset
        class_names (list): names of the classes in format alphabet/name/rotate

    Returns (list, list): list of paths to the classes,
    list of stings of rotations codes
    """
    d = []
    rots = []
    for class_name in class_names:
        alphabet, character, rot = class_name.split('/')
        image_dir = os.path.join(data_dir, 'data', alphabet, character)
        d.append(image_dir)
        rots.append(rot)
    return d, rots


def get_class_images_paths(dir_paths, rotates):
    """
    Return class names, paths to the corresponding images and rotations from
    the path of the classes' directories.

    Args:
        dir_paths (list): list of the class directories
        rotates (list): list of stings of rotation codes.

    Returns (list, list, list): list of class names, list of lists of paths to
    the images, list of rotation angles (0..240) as integers.

    """
    classes, img_paths, rotates_list = [], [], []
    for dir_path, rotate in zip(dir_paths, rotates):
        class_images = sorted(glob.glob(os.path.join(dir_path, '*.png')))

        classes.append(dir_path)
        img_paths.append(class_images)
        rotates_list.append(int(rotate[3:]))
    return classes, img_paths, rotates_list


def load_and_preprocess_image(img_path, rot):
    """
    Load and return preprocessed image.
    Args:
        img_path (str): path to the image on disk.
    Returns (Tensor): preprocessed image
    """
    img = Image.open(img_path).resize((28, 28)).rotate(rot)
    img = np.asarray(img)
    img = 1 - img
    return np.expand_dims(img, -1)


def load_omniglot(data_dir, config, splits):
    """
    Load omniglot dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    split_dir = os.path.join(data_dir, 'splits', config['data.split'])
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

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths and rotation angles per each class
        class_paths, rotates = class_names_to_paths(data_dir,
                                                    class_names)
        classes, img_paths, rotates = get_class_images_paths(
            class_paths,
            rotates)

        data = np.zeros([len(classes), len(img_paths[0]), 28, 28, 1])
        for i_class in range(len(classes)):
            for i_img in range(len(img_paths[i_class])):
                data[i_class, i_img, :, :,:] = load_and_preprocess_image(
                            img_paths[i_class][i_img], rotates[i_class])

        data_loader = OmniglotDataLoader(data,
                                         batch=config['data.batch'],
                                 n_classes=len(classes),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader
    return ret
