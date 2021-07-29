from imdb import IMDb
from collecting_and_preprocessing import config
import pandas as pd


PROCESSED_DATA_PATH = config.processed_data_path
RAW_DATA_PATH = config.data_movies_small


def main():
    movies_data = pd.read_csv(f'{RAW_DATA_PATH}/links.csv')
    imdbIds = movies_data['imdbId'].values
    movieIds = movies_data['movieId'].values

    ia = IMDb()
    movie_plots = []

    for imdbId, mId in zip(imdbIds, movieIds):
        node = {}
        try:
            movie = ia.get_movie(imdbId)
            node['movieId'] = mId
            node['plot'] = movie['plot']
            movie_plots.append(node)
        except:
            print(f'Exception occurred for movie: {imdbId}')

    movie_plots_data = pd.DataFrame(movie_plots)
    movie_plots_data.to_csv(f'{PROCESSED_DATA_PATH}/movie_plots_imdb.csv', index=False)


if __name__ == '__main__':
    main()
