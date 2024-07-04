# Apparel-Recommendation-Engine-Engine

# README

## Project Overview
This project demonstrates the application of various technical indicators and visualization tools to analyze and recommend apparel items based on a combination of textual descriptions, brand, color, product type, and image features. The main focus is on leveraging Word2Vec models to generate meaningful recommendations and visualize the similarities using heat maps.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Features](#features)
- [Technical Indicators and Visualization Tools](#technical-indicators-and-visualization-tools)
- [Contact](#contact)

## Installation
To run this project, you'll need to have Python installed. Additionally, you'll need to install the following libraries:

```bash
pip install PIL matplotlib numpy pandas beautifulsoup4 nltk seaborn keras gensim scikit-learn plotly
```

## Data
The dataset includes apparel information, such as titles, brands, colors, product types, and images. The data is preprocessed and stored in pickle and numpy files.

- `16k_data_cnn_features.npy`: CNN features of apparel images.
- `16k_data_cnn_feature_asins.npy`: ASINs corresponding to the CNN features.
- `pickels/16k_apperal_data_preprocessed`: Preprocessed apparel dataset.

## Usage
To use the project, follow these steps:

1. Ensure all necessary data files are in place.
2. Import the required libraries.
3. Load the Word2Vec model and dataset features.
4. Run the recommendation system and visualize the results.

```python
# Importing necessary libraries
import PIL.Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout
from IPython.display import display, Image, SVG, Math, YouTubeVideo

# Loading models and features
with open('word2vec_model', 'rb') as handle:
    model = pickle.load(handle)

bottleneck_features_train = np.load('16k_data_cnn_features.npy')
asins = np.load('16k_data_cnn_feature_asins.npy')
asins = list(asins)

data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
df_asins = list(data['asin'])

# Preprocessing
data['brand'].fillna(value="Not given", inplace=True)
brands = [x.replace(" ", "-") for x in data['brand'].values]
types = [x.replace(" ", "-") for x in data['product_type_name'].values]
colors = [x.replace(" ", "-") for x in data['color'].values]

# Vectorizing
idf_title_vectorizer = CountVectorizer()
idf_title_features = idf_title_vectorizer.fit_transform(data['title'])
brand_vectorizer = CountVectorizer()
brand_features = brand_vectorizer.fit_transform(brands)
type_vectorizer = CountVectorizer()
type_features = type_vectorizer.fit_transform(types)
color_vectorizer = CountVectorizer()
color_features = color_vectorizer.fit_transform(colors)
extra_features = hstack((brand_features, type_features, color_features)).tocsr()

# Calculate IDF values
idf_title_features = idf_title_features.astype(np.float)
for i in idf_title_vectorizer.vocabulary_.keys():
    idf_val = idf(i)
    for j in idf_title_features[:, idf_title_vectorizer.vocabulary_[i]].nonzero()[0]:
        idf_title_features[j, idf_title_vectorizer.vocabulary_[i]] = idf_val

# Load vocab
vocab = model.keys()

# Run the recommendation
idf_w2v_comp(12566, 5, 5, 10, 20)
```

## Features
- **Word2Vec Models**: Used to create vector representations of words in the apparel titles.
- **IDF Weighting**: Enhances the word vectors using inverse document frequency.
- **Visualization**: Heat maps and tables to visualize the similarities between apparel items.
- **Image Display**: Displays the recommended apparel images alongside the heat maps.

## Technical Indicators and Visualization Tools
The project utilizes various technical indicators and visualization tools to enhance the analysis:

- **PIL**: For image processing.
- **Matplotlib & Seaborn**: For plotting and heat maps.
- **NumPy & Pandas**: For data manipulation and analysis.
- **BeautifulSoup**: For web scraping.
- **NLTK**: For natural language processing.
- **Keras**: For building and training neural networks.
- **Gensim**: For working with Word2Vec models.
- **Scikit-learn**: For machine learning and vectorization.
- **Plotly**: For interactive visualizations.
