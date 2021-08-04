import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.constraints import non_neg

from .BaseCFModel import BaseCFModel


class CustomEmbeddingRegularizer(keras.layers.Layer):
    def __init__(self, tfidf_matrix, rate=4 * 1e-2):
        super(CustomEmbeddingRegularizer, self).__init__()
        self.rate = rate
        self.tfidf_matrix = tfidf_matrix
        self.map = self.calculate_map()

    def call(self, inputs, **kwargs):
        return self.rate * self.similar_embeddings(inputs)

    def calculate_map(self):
        mapping = {}
        for i in range(len(self.tfidf_matrix)):
            mapping[i] = []
            for j in range(len(self.tfidf_matrix)):
                if i != j:
                    dot_prod = abs(np.dot(self.tfidf_matrix[i], self.tfidf_matrix[j]))
                    if dot_prod >= 0.8:
                        mapping[i].append(j)
        return mapping

    def similar_embeddings(self, inputs):
        return tf.reduce_sum(tf.square(inputs))-tf.reduce_sum([tf.reduce_sum(
            tf.matmul(inputs[i, tf.newaxis],
                      tf.transpose(tf.gather(inputs, self.map[i], axis=0)))) / len(self.map[i])
         for i in range(len(inputs)) if len(self.map[i]) !=0])


class ContentBoostedCFModel(BaseCFModel):
    def __init__(self, N_USERS, N_MOVIES, LATENT_FACTORS, reg_term=1e-6, tf_idf_matrix=None):
        super().__init__(N_USERS, N_MOVIES, LATENT_FACTORS, reg_term, tf_idf_matrix)



    def build(self):
        movie_input = keras.layers.Input(shape=[1], name="MoviesInput")
        movie_embedding = keras.layers.Embedding(self.N_MOVIES,
                                                 self.LATENT_FACTORS,
                                                 name='MoviesEmbedding',
                                                 embeddings_regularizer=CustomEmbeddingRegularizer(self.tf_idf_matrix,
                                                                                                   rate=0.4),
                                                 embeddings_constraint=non_neg())(movie_input)
        movie_vec = keras.layers.Flatten(name='MoviesFlatten')(movie_embedding)

        user_input = keras.layers.Input(shape=[1], name='UsersInput')
        user_embedding = keras.layers.Embedding(self.N_USERS,
                                                self.LATENT_FACTORS,
                                                name='UsersEmbedding',
                                                embeddings_regularizer=keras.regularizers.l2(self.reg_term),
                                                embeddings_constraint=non_neg())(user_input)
        user_vec = keras.layers.Flatten(name='UsersFlatten')(user_embedding)

        product = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
        model = keras.Model([user_input, movie_input], product, name='MatrixFactorizationReccomender')
        self.model = model