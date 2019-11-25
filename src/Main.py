import numpy as np
import pandas as pd
from wordcloud1 import Wordcloudz
from word_processing import Word_Processing1
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import re
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  
from datalook import Datalook
from files import Files
from local_time import LocalTime
from textblob import TextBlob


from wordcloud import WordCloud, STOPWORDS

print('\n' * 10)
t = LocalTime()
print("=========================================================")
print("Local current time started :", t.localtime)
print("=========================================================")
twitter_file = "auspol2019.csv"
f1 = Files(twitter_file)
print(LocalTime().localtime, twitter_file, " read")
twitter = pd.read_csv(f1.file_path, parse_dates=['created_at','user_created_at'])
geocode_file = "location_geocode.csv"
f2 = Files(geocode_file)
print(LocalTime().localtime, geocode_file, " read")
geocodes = pd.read_csv(f2.file_path)
twitter = twitter.merge(geocodes, how='inner', left_on='user_location', right_on='name')
twitter = twitter.drop('name',axis =1)  # 'name' is a duplicate of 'user location' so remove.
print(LocalTime().localtime, "files merged")
#data = Datalook(twitter)
#data.show()

#Wordcloudz.show(twitter, 'user_description')
twitter['sentiment'] = twitter['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
print(LocalTime().localtime, "5 random tweets with highest positive sentiment polarity: \n")
cL = twitter.loc[twitter.sentiment==1, ['full_text']].sample(5).values
positive_sentences = []
for c in cL:
    print(c[0])
    print()

cL = twitter.loc[twitter.sentiment==1, ['full_text']].values
for c in cL:
    words = c[0].split()
    words = map(lambda x: Word_Processing1.clean_word(x), words) # Remove "stop" words that do not influence sentiment
    words = list(filter(lambda x:True if len(x) > 0 else False, words))
    positive_sentences.append(words)

print("-----------------------")
print(LocalTime().localtime, "number of positive sentiments = ", len(positive_sentences))

negative_sentences = []
cL = twitter.loc[twitter.sentiment==0, ['full_text']].values
for c in cL:
    words = c[0].split()
    words = map(lambda x: Word_Processing1.clean_word(x), words) # Remove "stop" words that do not influence sentiment
    words = list(filter(lambda x:True if len(x) > 0 else False, words))
    negative_sentences.append(words)

print("-----------------------")
print(LocalTime().localtime, "number of negative sentiments = ", len(negative_sentences))

t = LocalTime()
print("=========================================================")
print("Local current time completed :", t.localtime)
print("=========================================================")
