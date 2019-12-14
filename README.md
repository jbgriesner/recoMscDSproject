# Recommender Systems Project

## 0. Quick Start
To run this notebook you just need to have [pipenv](https://github.com/pypa/pipenv) installed.
Then run these 3 commands:
- first install the dependencies with: `pipenv install`
- launch the virtual env: `pipenv shell`
- finally start jupyter and open the notebook: `jupyter-lab`


```python
import sys
sys.path.append("../src")
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

from surprise import NormalPredictor, SVD, KNNBasic, NMF
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, KFold
```

## 1. Introduction
Recommender systems goal is to push *relevant* items to a given user. Understanding and modelling the user's preferences is required to reach this goal. In this project you will learn how to model the user's preferences with the [Surprise library](http://surpriselib.com/) to build different recommender systems. The first one will be a pure *collaborative filtering* approach, and the second one will rely on item attributes in a *content-based* way.

## 2. Loading Data
We use here the [MovieLens dataset](https://grouplens.org/datasets/movielens/). It contains 25 millions of users ratings. the data are in the `./data/raw` folder. We could load directly the .csv file with [a built-in Surprise function](https://github.com/NicolasHug/Surprise/blob/ef3ed6e98304dbf8d033c8eee741294b05b5ba07/surprise/dataset.py#L105), but it's more convenient to load it through a Pandas dataframe for later flexibility purpose.


```python
RATINGS_DATA_FILE = './data/raw/ratings.csv'
MOVIES_DATA_FILE = './data/raw/movies.csv'
```


```python
# load the raw csv into a data_frame
df_ratings = pd.read_csv(RATINGS_DATA_FILE)

# drop the timestamp column since we dont need it now
df_ratings = df_ratings.drop(columns="timestamp")

# movies dataframe
df_movies = pd.read_csv(MOVIES_DATA_FILE)
```


```python
# check we have 25M users' ratings
df_ratings.userId.count()
```




    25000095




```python
def get_subset(df, number):
    """
        just get a subset of a large dataset for debug purpose
    """
    rids = np.arange(df.shape[0])
    np.random.shuffle(rids)
    df_subset = df.iloc[rids[:number], :].copy()
    return df_subset
df_ratings_100k = get_subset(df_ratings, 100000)
df_movies_100 = get_subset(df_movies, 100)
```


```python
# Surprise reader
reader = Reader(rating_scale=(0, 5))

# Finally load all ratings
ratings = Dataset.load_from_df(df_ratings_10k, reader)
```

## 3. Collaborative Filtering
We can test first any of the [Surprise algorithms](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html).


```python
# define a cross-validation iterator
kf = KFold(n_splits=3)

algos = [SVD(), NMF(), KNNBasic()]    
```


```python
def get_rmse(algo, testset):
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)
        
for trainset, testset in tqdm(kf.split(ratings)): 
    """
        get an evaluation with cross-validation for different algorithms
    """  
    for algo in algos:
        algo.fit(trainset)
        get_rmse(algo, testset)
```

    0it [00:00, ?it/s]

    RMSE: 1.0408
    RMSE: 1.0949
    Computing the msd similarity matrix...


    1it [00:01,  1.57s/it]

    Done computing similarity matrix.
    RMSE: 1.0608
    RMSE: 1.0489
    RMSE: 1.1087
    Computing the msd similarity matrix...


    2it [00:03,  1.56s/it]

    Done computing similarity matrix.
    RMSE: 1.0745
    RMSE: 1.0303
    RMSE: 1.0855
    Computing the msd similarity matrix...


    3it [00:04,  1.54s/it]

    Done computing similarity matrix.
    RMSE: 1.0559


    


**TODO from now**: 
- test different similarity measures
- test different computation methods (ALS vs SGD) and conclude
- test with different parameters
- tune to find the best parameters
- visualization & interpretation of results
- these first results can serve as a baseline to improve next
- personalize candidate generation by selecting the most popular items  

## 4. Content-based Filtering
Here we will rely directly on items attributes. First we have to describe a user profile with an attributes vector. Then we will use these vectors to generate recommendations.


```python
# computing similarities requires too much ressources on the whole dataset, so we take the subset with 100 items
df_movies_100 = df_movies_100.reset_index(drop=True)
df_movies_100.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>162126</td>
      <td>Autobiography of a Princess (1975)</td>
      <td>(no genres listed)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>194666</td>
      <td>Roads in February (2018)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>157679</td>
      <td>Alley Cats Strike (2000)</td>
      <td>Children|Comedy|Drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>169196</td>
      <td>Once Upon a Time Veronica (2012)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>191777</td>
      <td>Revenge: A Love Story (2010)</td>
      <td>Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we compute a TFIDF on the titles of the movies
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df_movies_100['title'])
```


```python
# we get cosine similarities: this takes a lot of time on the real dataset
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
```


```python
# we generate in 'results' the most similar movies for each movie: we put a pair (score, movie_id)
results = {}
for idx, row in df_movies_100.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
    similar_items = [(cosine_similarities[idx][i], df_movies_100['movieId'].loc[[i]].tolist()[0]) for i in similar_indices] 
    results[idx] = similar_items[1:]
```


```python
len(results)
```




    100




```python
# transform a 'movieId' into its corresponding movie title
def item(id):  
    return df_movies_100.loc[df_movies_100['movieId'] == id]['title'].tolist()[0].split(' - ')[0] 
```


```python
# transform a 'movieId' into the index id
def get_idx(id):
    return df_movies_100[df_movies_100['movieId'] == id].index.tolist()[0]
```


```python
# Finally we put everything together here:
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")   
    print("-------")    
    recs = results[get_idx(item_id)][:num]   
    for rec in recs: 
        print("\tRecommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")
```

Suppose a user wants the 10 most 'similar' (from a CBF point of view) movies from the movie 'Alley Cats Strike':


```python
recommend(item_id=157679, num=10)
```

    Recommending 10 products similar to Alley Cats Strike (2000)...
    -------
    	Recommended: Ringu 0: Bâsudei (2000) (score:0.10424703060511913)
    	Recommended: 6th Day, The (2000) (score:0.10424703060511913)
    	Recommended: Room 205 of Fear (2011) (score:0.0)
    	Recommended: Legend (2015) (score:0.0)
    	Recommended: Hardcore (2001) (score:0.0)
    	Recommended: The Huntress: Rune of the Dead (2019) (score:0.0)
    	Recommended: House of Dracula (1945) (score:0.0)
    	Recommended: Schramm (1993) (score:0.0)
    	Recommended: The Coed and the Zombie Stoner (2014) (score:0.0)
    	Recommended: Honor Among Lovers (1931) (score:0.0)


**TODO**:
- what are the advantages of CBF over CF ? Discuss results...
- Surprise does not support content-based information. The goal here is to implement a content-based algorithm in to Surprise


