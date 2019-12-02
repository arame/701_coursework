import numpy as np
import pandas as pd
from wordcloud1 import Wordcloudz
from word_processing import Word_Processing1
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  
from datalook import Datalook
from files import Files
from local_time import LocalTime
from sentences1 import Sentences1
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

print('\n' * 10)
print("=========================================================")
print("Local current time started :", LocalTime.get())
print("=========================================================")
twitter_file = "auspol2019.csv"
f1 = Files(twitter_file)
print(LocalTime.get(), twitter_file, " read")
twitter = pd.read_csv(f1.file_path, parse_dates=['created_at','user_created_at'])
geocode_file = "location_geocode.csv"
f2 = Files(geocode_file)
print(LocalTime.get(), geocode_file, " read")
geocodes = pd.read_csv(f2.file_path)
twitter = twitter.merge(geocodes, how='inner', left_on='user_location', right_on='name')
twitter = twitter.drop('name',axis =1)  # 'name' is a duplicate of 'user location' so remove.
twitter['sentiment'] = twitter['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
#twitter1 = [x for x in twitter if x['sentiment'] == 0]
twitter1 = []
i = -1
for x in twitter['sentiment']:
    i = i + 1
    if x != 0:
        twitter1.append(twitter[i])
# TODO fix this
target = []
i = -1
for t in twitter1['sentiment']:
    i = i + 1
    if t > 0:
        target.append(1)
    else:
        target.append(0)

#target = twitter['sentiment'](filter(lambda x:math.ceil(x) if x > 0 else math.floor(x), target))
#words = list(filter(lambda x:True if len(x) > 0 else False, words))
print(LocalTime.get(), "files merged")
####### split the dataset in 2, 80% as training data and 20% as testing data
train, test = train_test_split(twitter, test_size=0.2)
print("-----------------------")
print("Train data = ", train.shape)
print("Test data = ", test.shape)
print("-----------------------")
######## Vectorise the train and test datasets
cv = CountVectorizer(binary=True)
cv.fit(train)
X = cv.transform(train)
X_test = cv.transform(test)
######## Build the classifiers
twitter_rows = twitter.shape[0]
print("twitter number of rows = ", twitter_rows)

#X_train, X_val, y_train, y_val = train_test_split(
#    X, target, train_size = 0.75
#)
#data = Datalook(twitter)
#data.show()

#Wordcloudz.show(twitter, 'user_description')
twitter['sentiment'] = twitter['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
print(LocalTime.get(), "5 random tweets with highest positive sentiment polarity: \n")
cL = twitter.loc[twitter.sentiment==1, ['full_text']].sample(5).values
for c in cL:
    print(c[0])
    print()

cL = twitter.loc[twitter.sentiment >0, ['full_text']].values
positive_sentences = Sentences1.filter(cL)
number_positive = len(positive_sentences)
percentage_positive = "which is {0:.2f}% of all tweets".format((number_positive / len(twitter)) * 100)
print("-----------------------")
print(LocalTime.get(), "Number of positive sentiments = ", number_positive, percentage_positive)

cL = twitter.loc[twitter.sentiment < 0, ['full_text']].values
negative_sentences = Sentences1.filter(cL)
number_negative = len(negative_sentences)
percentage_negative = "which is {0:.2f}% of all tweets".format((number_negative / len(twitter)) * 100)
print("-----------------------")
print(LocalTime.get(), "Number of negative sentiments = ", number_negative, percentage_negative)

cL = twitter.loc[twitter.sentiment==0, ['full_text']].values
neutral_sentences = Sentences1.filter(cL)
number_neutral = len(neutral_sentences)
percentage_neutral = "which is {0:.2f}% of all tweets".format((number_neutral / len(twitter)) * 100)
print("-----------------------")
print(LocalTime.get(), "Number of neutral sentiments = ", number_neutral, percentage_neutral)
print("-----------------------")

count_of_tweets = len(twitter)
count_of_retweets = np.sum(twitter.retweet_count)
print(f"Total number of tweets = ", count_of_tweets)
print(f"Total number of retweets = ", count_of_retweets)
print(f"Average number of retweets per tweet = ", count_of_retweets / count_of_tweets)
tweets_retweeted = twitter.apply(lambda x:True if x["retweet_count"] > 0 else False, axis = 1)
count_of_tweets_retweeted = len(tweets_retweeted[tweets_retweeted == True])
print(f"% of tweets retweeted = ", ( count_of_tweets_retweeted / count_of_tweets) * 100)
print(f"Number of tweets retweeted = ", count_of_tweets_retweeted)
print(f" Maximum number of retweets {twitter.retweet_count.max()}")
print(f" Maximum number of favorites {twitter.favorite_count.max()}")
print("-----------------------")
print(f"Most retweeted: ", twitter.loc[twitter['retweet_count']==6622.0,'full_text'].values)
print("-----------------------")
print(f"Most favourited: ",twitter.loc[twitter['favorite_count']==15559.0,['full_text','user_name','user_description']].values)
print("=========================================================")
print("Local current time completed :", LocalTime.get())
print("=========================================================")
