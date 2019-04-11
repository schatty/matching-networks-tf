import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MatchingNetwork(Model):
    def __init__(self):
        super(MatchingNetwork, self).__init__()

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

        #self.calc_dist = CosineDistance()

    @tf.function
    def call(self, x_support, y_support, x_query, y_query):

        #print("Support: ", x_support.shape, y_support.shape)
        #print("Query: ", x_query.shape, y_query.shape)
        #print("Type: ", type(y_support))

        y_support = tf.cast(y_support, tf.int32)
        y_hot = tf.one_hot(y_support, 5, axis=-1)
        y_support = tf.cast(y_hot, tf.float32)

        y_query = tf.cast(y_query, tf.int32)
        y_query_one_hot = tf.one_hot(y_query, 5, axis=-1)
        y_query_one_hot = tf.cast(y_query_one_hot, tf.float32)

        # Embeddings for support images
        emb_imgs = []
        for i in range(25):
            emb_imgs.append( self.g(x_support[:, i, :, :, :]) )

        # Embeddings for query images
        for i in range(5):
            query_emb = self.g(x_query[:, i, :, :, :])
            emb_imgs.append(query_emb)
            outputs = tf.stack(emb_imgs)

            # Cosine similarity between support set and query
            
            x = outputs[:-1]
            y = outputs[-1]
            eps = 1e-10
            similarities = tf.zeros([25, 32], tf.float32)
            print("Similarities: ", similarities)
            j = 0
            for support_image in x:
                sum_support = tf.reduce_sum(tf.square(support_image), axis=1)
                support_magnitude = tf.math.rsqrt(
                    tf.clip_by_value(sum_support, eps, float("inf")))
                dot_prod = tf.keras.backend.batch_dot(
                    tf.expand_dims(y, 1),
                    tf.expand_dims(support_image, 2)
                )
                dot_prod = tf.squeeze(dot_prod)
                print("Dot Prod: ", dot_prod.shape)
                cos_sim = tf.multiply(dot_prod, support_magnitude)
                print("Cos sim: ", cos_sim.shape)
                cos_sim = tf.reshape(cos_sim, [1, -1])
                similarities = tf.tensor_scatter_nd_update(similarities,
                                            [[j]],
                                            cos_sim)
                j += 1

            similarities = tf.transpose(similarities)
            print("Similarities: ", similarities.shape)

            # Produce predictions for target probabilities
            similarities = tf.nn.softmax(similarities)
            similarities = tf.expand_dims(similarities, 1)
            print("sim softed and unqueezed: ", similarities.shape)
            preds = tf.keras.backend.batch_dot(similarities, y_support)
            preds = tf.squeeze(preds)
            print("preds: ", preds.shape)

            print("Final shapes: ", tf.squeeze(y_query_one_hot[:, i, :]).shape,
                  tf.math.log(preds).shape)
            #tf.print(y_query_one_hot[:, i, :])

            if i == 0:
                ce = tf.keras.backend.categorical_crossentropy(y_query_one_hot[:, i, :], preds)
                eq = tf.cast(tf.equal(
                    tf.cast(tf.argmax(preds, axis=-1), tf.int32),
                    tf.cast(y_query[:, i], tf.int32)), tf.float32)
                acc = tf.reduce_mean(eq)

            else:
                ce += tf.keras.backend.categorical_crossentropy(y_query_one_hot[:, i, :], preds)
                eq = tf.cast(tf.equal(
                    tf.cast(tf.argmax(preds, axis=-1), tf.int32),
                    tf.cast(y_query[:, i], tf.int32)), tf.float32)
                acc += tf.reduce_mean(eq)
        print("Return ")
        return ce/5, acc/5

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass