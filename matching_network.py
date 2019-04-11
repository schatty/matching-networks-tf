import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional
from tensorflow.keras import Model
from tensorflow.keras.backend import categorical_crossentropy, batch_dot


class MatchingNetwork(Model):
    def __init__(self, way):
        super(MatchingNetwork, self).__init__()

        self.way = way
        self.g = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )
        # Fully contextual embedding
        self.fce = tf.keras.Sequential([
            Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        ])

    @tf.function
    def call(self, x_support, y_support, x_query, y_query):
        def _calc_cosine_distances(support, query_img):
            eps = 1e-10
            similarities = tf.zeros([self.support_samples, self.batch],
                                    tf.float32)
            i_sample = 0
            for support_image in support:
                sum_support = tf.reduce_sum(tf.square(support_image), axis=1)
                support_magnitude = tf.math.rsqrt(
                    tf.clip_by_value(sum_support, eps, float("inf")))
                dot_prod = batch_dot(
                    tf.expand_dims(query_img, 1),
                    tf.expand_dims(support_image, 2)
                )
                dot_prod = tf.squeeze(dot_prod)
                cos_sim = tf.multiply(dot_prod, support_magnitude)
                cos_sim = tf.reshape(cos_sim, [1, -1])
                similarities = tf.tensor_scatter_nd_update(similarities,
                                                           [[i_sample]],
                                                           cos_sim)
                i_sample += 1
            return tf.transpose(similarities)

        self.batch = x_support.shape[0]
        self.support_samples = x_support.shape[1]
        self.query_samples = x_query.shape[1]

        y_support = tf.cast(y_support, tf.int32)
        y_support_one_hot = tf.one_hot(y_support, self.way, axis=-1)
        y_support_one_hot = tf.cast(y_support_one_hot, tf.float32)

        y_query = tf.cast(y_query, tf.int32)
        y_query_one_hot = tf.one_hot(y_query, self.way, axis=-1)
        y_query_one_hot = tf.cast(y_query_one_hot, tf.float32)

        # Embeddings for support images
        emb_imgs = []
        for i in range(self.support_samples):
            emb_imgs.append( self.g(x_support[:, i, :, :, :]) )

        # Embeddings for query images
        for i_query in range(self.query_samples):
            query_emb = self.g(x_query[:, i_query, :, :, :])
            emb_imgs.append(query_emb)
            outputs = tf.stack(emb_imgs)

            # Fully contextual embedding
            outputs = self.fce(outputs)

            # Cosine similarity between support set and query
            similarities = _calc_cosine_distances(outputs[:-1], outputs[-1])

            # Produce predictions for target probabilities
            similarities = tf.nn.softmax(similarities)
            similarities = tf.expand_dims(similarities, 1)
            preds = tf.squeeze(batch_dot(similarities, y_support_one_hot))

            query_labels = y_query_one_hot[:, i_query, :]
            eq = tf.cast(tf.equal(
                tf.cast(tf.argmax(preds, axis=-1), tf.int32),
                tf.cast(y_query[:, i_query], tf.int32)), tf.float32)
            if i_query == 0:
                ce = categorical_crossentropy(query_labels, preds)
                acc = tf.reduce_mean(eq)

            else:
                ce += categorical_crossentropy(query_labels, preds)
                acc += tf.reduce_mean(eq)

            emb_imgs.pop()

        return ce/self.query_samples, acc/self.query_samples

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass