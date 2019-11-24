import numpy as np
import pandas as pd
from wordcloud1 import Wordcloudz
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
import re
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  
from datalook import Datalook
from files import Files
from local_time import LocalTime
from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

from wordcloud import WordCloud, STOPWORDS

print('\n' * 10)
t = LocalTime()
print("=========================================================")
print("Local current time started :", t.localtime)
print("=========================================================")
twitter_file = "auspol2019.csv"
f1 = Files(twitter_file)
twitter = pd.read_csv(f1.file_path, parse_dates=['created_at','user_created_at'])
geocode_file = "location_geocode.csv"
f2 = Files(geocode_file)
geocodes = pd.read_csv(f2.file_path)
twitter = twitter.merge(geocodes, how='inner', left_on='user_location', right_on='name')
twitter = twitter.drop('name',axis =1)  # 'name' is a duplicate of 'user location' so remove.
#data = Datalook(twitter)
#data.show()
#stopword_file = 'long_stopwords.txt'
#f3 = Files(stopword_file)
#with open(f3.file_path,'r') as inpFile:
    #lines = inpFile.readlines()
    #stop_words_temp = map(lambda x : re.sub('\n','',x),lines)
    #stop_words = list(map(lambda x:  re.sub('[^A-Za-z0-9]+', '',x), stop_words_temp))
    #print(stop_words)

#Wordcloudz.show(twitter, 'user_description')
print("Stop words")
print(eng_stopwords)
twitter['sentiment'] = twitter['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
print("5 random tweets with highest positive sentiment polarity: \n")
cL = twitter.loc[twitter.sentiment==0, ['full_text']].sample(5).values
for c in cL:
    print(c[0])
    print()

t = LocalTime()
print("=========================================================")
print("Local current time completed :", t.localtime)
print("=========================================================")
