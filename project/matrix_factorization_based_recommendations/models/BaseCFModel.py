import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.constraints import non_neg
from IPython.display import SVG, display
from tensorflow.keras.utils import model_to_dot

class BaseCFModel:

  model = None

  def __init__(self, N_USERS, N_MOVIES, LATENT_FACTORS, reg_term=1e-6, tf_idf_matrix=None):
    self.N_USERS = N_USERS
    self.N_MOVIES = N_MOVIES
    self.LATENT_FACTORS = LATENT_FACTORS
    self.tf_idf_matrix = tf_idf_matrix
    self.reg_term = reg_term
    self.build()

  def build(self):
      movie_input = keras.layers.Input(shape=[1], name="MoviesInput")
      movie_embedding = keras.layers.Embedding(self.N_MOVIES,
                                               self.LATENT_FACTORS,
                                               name='MoviesEmbedding',
                                               embeddings_regularizer=keras.regularizers.l2(self.reg_term),
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
      model.run_eagerly = True
      self.model = model

  def compile(self, loss_function='mean_squared_error', lr=0.01):
      opt = keras.optimizers.SGD(learning_rate=lr)
      self.model.compile(optimizer=opt, loss=loss_function, metrics=['mae', 'mse'])
      print(self.model.summary())
      display(SVG(model_to_dot(self.model,
                               show_shapes=True,
                               show_layer_names=True,
                               rankdir='HB',
                               dpi=70)
                  .create(prog='dot', format='svg')))

  def train(self, train_data, epochs=100, batch=8):
      history = self.model.fit([train_data.userId, train_data.movieId],
                               train_data.rating,
                               epochs=epochs,
                               batch_size=batch,
                               validation_split=0.15,
                               verbose=1)

      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('Training loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'])
      plt.show()

  def get_movies_embeddings(self, index=None):
      movie_embeddings = self.model.get_layer(name='MoviesEmbedding').get_weights()[0]
      if index is None:
          return movie_embeddings
      else:
          return movie_embeddings[index]

  def recommend_movies(self, user_id, num_movies = 1):
      user_embedding_learnt = self.model.get_layer(name='UsersEmbedding').get_weights()[0][user_id]
      movie_embedding_learnt = self.get_movies_embeddings()
      movies = user_embedding_learnt[user_id] @ movie_embedding_learnt.T
      mids = np.argpartition(movies, -num_movies)[-num_movies:]
      return mids

  def predict(self, data):
      return self.model.predict([data.userId, data.movieId])

  def evaluate(self, pred, true):
      return self.model.evaluate(pred, true)

  def save(self, path):
      self.model.save(path)

  def load_model(self, path):
      self.model = keras.models.load_model(path)
      print(self.model.summary())



