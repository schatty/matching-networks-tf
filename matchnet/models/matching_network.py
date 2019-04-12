import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, \
    BatchNormalization, ReLU, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.backend import categorical_crossentropy, batch_dot


class MatchingNetwork(Model):
    def __init__(self, way, w, h, c, lstm_size=32, batch_size=32):
        super(MatchingNetwork, self).__init__()

        self.way = way
        self.w, self.h, self.c = w, h, c
        self.batch_size = batch_size

        self.g = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),
            Flatten()]
        )
        # Fully contextual embedding
        self.fce_dim = lstm_size * 2
        self.fce = tf.keras.Sequential([
            Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        ])

    @tf.function
    def call(self, x_support, y_support, x_query, y_query):
        def _calc_cosine_distances(support, query_img):
            """
            Calculate cosine distances between support images and query one.
            Args:
                support (Tensor): Tensor of support images
                query_img (Tensor): query image

            Returns:

            """
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

        # Get one-hot representation
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

    def save(self, dir_path):
        """
        Save model to the provided directory.

        Args:
            dir_path (str): path to the directory to save the model files.

        Returns: None

        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Save CNN encoder
        self.g.save(os.path.join(dir_path, 'cnn_encoder.h5'))
        # Save LSTM
        self.fce.save(os.path.join(dir_path, 'lstm.h5'))

    def load(self, dir_path):
        """
        Load model from provided directory.

        Args:
            dir_path (str): path to the directory from where restore model.

        Returns: None

        """
        # Encoder CNN
        encoder_path = os.path.join(dir_path, 'cnn_encoder.h5')
        self.g(tf.zeros([1, self.w, self.h, self.c]))
        self.g.load_weights(encoder_path)
        # LSTM
        lstm_path = os.path.join(dir_path, 'lstm.h5')
        self.fce(tf.zeros([1, self.batch_size, self.fce_dim]))
        self.fce.load_weights(lstm_path)