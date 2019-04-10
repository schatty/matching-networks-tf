import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MatchingNetwork(Model):
    def __init__(self):
        super(MatchingNetwork, self).__init__()

    def call(self, x_support, y_support, x_query, y_query):
        print("Support: ", x_support.shape, y_support.shape)
        print("Query: ", x_query.shape, y_query.shape)
        loss = 0
        acc = 0
        return loss, acc

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass