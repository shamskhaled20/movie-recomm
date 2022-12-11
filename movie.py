#!/usr/bin/env python
# coding: utf-8

# In[31]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for jupyter notebook widgets
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual

# for Interactive Shells
from IPython.display import display


# In[32]:



# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[33]:


pd.options.display.max_columns = 50


# Read Data

# In[35]:


# lets read the dataset
data = pd.read_csv('E:/codeclause internship/project 2/movies_metadata.csv')
data.head()


# In[36]:


# lets check the shape
print(data.shape)


# In[37]:



# lets check the column wise info
data.info()


# In[38]:


data.isnull().sum()


# In[39]:


# check duplicated data
data.duplicated().sum()

# drop duplicates
data = data.drop_duplicates()


# In[40]:



data.nunique()


# In[41]:


data.describe()


# In[42]:


data.dtypes


# In[43]:


movie = data.copy()
movie = data.dropna()


# In[46]:


movie['release_date'] = pd.to_numeric(movie['release_date'], errors="coerce")


# Top 10 Movies with Highest IMBD Score
# 

# In[48]:


top10 = movie.sort_values(by='vote_average', ascending=False)
top10 = top10[['title','release_date', 'genres', 'overview', 'vote_average']].reset_index(drop=True)
top10 = top10.head(10)
top10


# In[50]:


sns.barplot(y='title', x='vote_average', data=top10, palette='Blues')
plt.title('Top 10 Movies with Highest IMBD Score', fontsize=12)


# Top 10 Movies with Highest Gross
# 

# In[52]:


top10gross = movie[['budget', 'title', 'release_date']].sort_values(by='budget', ascending=False).reset_index(drop=True).head(10)


# Movie Comparison by Language
# 

# In[61]:


movie['original_language'].value_counts()


# In[62]:


movie['original_language'] = np.where(movie['original_language'] == 'fr', 'en', 'Foreign')
movie['original_language'].value_counts()


# In[63]:


imdb_lang = movie.groupby('original_language').agg(imdb=('vote_average','mean')).reset_index()
imdb_lang


# In[64]:


data['runtime'].describe()


# In[65]:


def duration(x, quantile):
    q1, q3 = quantile
    try:
        if x >= q3:
            return 'long'
        elif x >= q1:
            return 'medium'
        else:
            return 'short'
    except exception:
        return 'Unknown'

movie['duration_type'] = movie['runtime'].apply(lambda x: duration(x, movie['runtime'].quantile([0.33, 0.66])))


# In[66]:


movie['duration_type'].value_counts()


# In[68]:


gross_duration = movie.groupby('duration_type').agg(gross_sum=('budget','sum'), gross_mean=('budget','mean')).reset_index()
gross_duration


# In[70]:


imdb_duration = movie.groupby('duration_type').agg(imdb=('vote_average','mean')).reset_index()
imdb_duration


# Genres
# 

# In[71]:


movie['genres'].value_counts()


# In[73]:


movie['genre'] = movie['genres'].str.split('|')
movie['genre1'] = movie['genre'].apply(lambda x: x[0])

# Some of the movies have only one genre. In such cases, assign the same genre to 'genre_2' as well
movie['genre2'] = movie['genre'].apply(lambda x: x[1] if len(x) > 1 else x[0])
movie['genre3'] = movie['genre'].apply(lambda x: x[2] if len(x) > 2 else x[0])
movie['genre4'] = movie['genre'].apply(lambda x: x[3] if len(x) > 3 else x[0])
movie['genre5'] = movie['genre'].apply(lambda x: x[5] if len(x) > 5 else x[0])

movie[['genres', 'genre1', 'genre2', 'genre3', 'genre4', 'genre5']].head(5)


# In[74]:


movie.head()


# In[79]:


gen = movie.groupby('genre1').agg(count=('genre1','count')).reset_index()
gen


# In[81]:


gross_genre = movie.groupby('genre1').agg(gross_sum=('budget','sum'),gross_mean=('budget','mean')).reset_index()
gross_genre


# Recommending similar Movies
# 

# In[83]:


# lets read the dataset
df = pd.read_csv('E:/codeclause internship/project 2/movies_metadata.csv')


# In[87]:


pip install mlxtend


# In[91]:


from mlxtend.preprocessing import TransactionEncoder

x = df['genres'].str.split('name')
te = TransactionEncoder()
x = te.fit_transform(x)
x = pd.DataFrame(x, columns = te.columns_)

# lets convert this data into boolean so that we can perform calculations
genres = x.astype('int')

# now, lets insert the movie titles in the first column, so that we can better understand the data
genres.insert(0, 'title', df['title'])

# lets set these movie titles as index of the data
genres = genres.set_index('title')

genres.head()


# In[93]:


# lets make a sparse matrix to recommend the movies

x = genres.transpose()
x.head()

