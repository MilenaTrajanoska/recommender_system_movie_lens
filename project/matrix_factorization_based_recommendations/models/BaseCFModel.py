import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np


class BaseCFModel:

  model = None

  def __init__(self, N_USERS, N_MOVIES, LATENT_FACTORS, metrics=None):
    self.N_USERS = N_USERS
    self.N_MOVIES = N_MOVIES
    self.LATENT_FACTORS = LATENT_FACTORS
    self.build()


  def build(self):
      movie_input = keras.layers.Input(shape=[1], name="MoviesInput")
      movie_embedding = keras.layers.Embedding(self.N_MOVIES + 1, self.LATENT_FACTORS, name='MoviesEmbedding')(movie_input)
      movie_vec = keras.layers.Flatten(name='MoviesFlatten')(movie_embedding)

      user_input = keras.layers.Input(shape=[1], name='UsersInput')
      user_embedding = keras.layers.Embedding(self.N_USERS + 1, self.LATENT_FACTORS, name='UsersEmbedding')(user_input)
      user_vec = keras.layers.Flatten(name='UsersFlatten')(user_embedding)

      product = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
      model = keras.Model([user_input, movie_input], product, name='MatrixFactorizationReccomender')
      self.model = model


  def compile(self, loss_function='mean_squared_error'):
      self.model.compile(optimizer='sgd', loss=loss_function, metrics=['mae', 'mse'])
      print(self.model.summary())
      tf.keras.utils.plot_model(self.model, to_file='model.png')


  def train(self, train_data, epochs = 100, batch = 8):
      history = self.model.fit([train_data.userId, train_data.movieId],
                               train_data.rating,
                               epochs=epochs,
                               batch_size=batch,
                               validation_split=0.15,
                               verbose=0)

      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Training loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'])
      plt.show()


  def get_movies_embeddings(self, index = None):
      movie_embeddings = self.model.get_layer(name='MoviesEmbedding').get_weights()[0]
      if index is None:
          return pd.DataFrame(movie_embeddings)
      else:
          return movie_embeddings[index]

  def recommend_movies(self, user_id, num_movies = 1):
      user_embedding_learnt = self.model.get_layer(name='UsersEmbedding').get_weights()[0][user_id]
      movie_embedding_learnt = self.get_movies_embeddings()
      movies = user_embedding_learnt[user_id] @ movie_embedding_learnt.T
      mids = np.argpartition(movies, -num_movies)[-num_movies:]
      return mids

  def save(self, folder='base'):
      self.model.save(f'../trained_models/{folder}')

  def load_model(self, path):
      self.model = keras.models.load_model(path)
      print(self.model.summary())



