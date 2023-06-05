import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

dt = pd.read_csv("/kaggle/input/music-dataset-song-information-and-lyrics/songs.csv")

sns.histplot(dt, x='Popularity', kde=True, color='g')

dt['Overview'] = dt['Artist'] + ". " + dt['Album'] + ". " + dt['Popularity'].astype(str) + ". " + dt['Lyrics']

dt['Overview'] = dt['Overview'].apply(lambda x: x.lower())

dt['Name'] = dt['Name'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(dt['Overview']).toarray()

similarity = cosine_similarity(vector)

def similar_song(name):
    name = name.lower()
    indices = dt[dt['Name'] == name].index[0]
    distances = similarity[indices]
    arr = sorted(list(enumerate(distances)), reverse = True, key=lambda x: x[1])[1:6]
    print("Recommended options are:")
    for i in arr:
        song_name = dt.loc[i[0], 'Name']
        artist = dt.loc[i[0], 'Artist']
        album = dt.loc[i[0],'Album']
        print("{} by {}, album - {}".format(song_name.capitalize(), artist, album))

similar_song("Imagine - Remastered 2010")