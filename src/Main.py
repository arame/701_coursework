import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud1 import Wordcloudz
from word_processing import Word_Processing1
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  
from datalook import Datalook
from files import Files
from logistic_regression import Logistic_Regression
from linear_svm import Linear_SVM
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

print("twitter number of rows = ", twitter.shape[0])
##### Remove neutral sentiments 
twitter1 = twitter[twitter.sentiment != 0]
####### Set targets to 1 for positive sentiment and 0 for negative sentiment

print(LocalTime.get(), "files merged and sentiment rating calculated")
####### split the dataset in 2, 80% as training data and 20% as testing data
train, test = train_test_split(twitter1, test_size=0.2)
train_text = Sentences1.filter(train['full_text'])
test_text = Sentences1.filter(test['full_text'])
target = np.where(train.sentiment > 0, 1, 0)
target_test = np.where(test.sentiment > 0, 1, 0)

print("-----------------------")
print("Train and test data divided")
print("Train data = ", train.shape)
print("Test data = ", test.shape)
print("-----------------------")
######## Vectorise the train and test datasets
cv = CountVectorizer(binary=True)
X = cv.fit_transform(train_text)
X_test = cv.transform(test_text)
######## Build the classifiers
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)
print('\n' * 2)
print("---------------------------------------------------")
print(LocalTime.get(), "  Words selected report")
print("---------------------------------------------------")
best_c = Logistic_Regression.get_best_hyperparameter(X_train, y_train, y_val, X_val)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
final_accuracy = final_model.predict(X_test)
final_accuracy_score = accuracy_score(target_test, final_accuracy)
print ("Final Accuracy: %s" % final_accuracy_score)
feature_names = zip(cv.get_feature_names(), final_model.coef_[0])
feature_to_coef = {
    word: coef for word, coef in feature_names
}
itemz = feature_to_coef.items()
list_positive = sorted(
    itemz, 
    key=lambda x: x[1], 
    reverse=True)[:5]
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular positve words")
for best_positive in list_positive:
    print (best_positive)
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular negative words")
list_negative = sorted(
    itemz, 
    key=lambda x: x[1])[:5]
for best_negative in list_negative:
    print (best_negative)

print('\n' * 2)
print("----------------------------------------------------------")
print(LocalTime.get(), "  Words selected report: SVM ")
print("----------------------------------------------------------")
best_c = Linear_SVM.get_best_hyperparameter(X_train, y_train, y_val, X_val)
final_svm  = LinearSVC(C=best_c)
final_svm.fit(X, target)
final_accuracy = final_svm.predict(X_test)
final_accuracy_score = accuracy_score(target_test, final_accuracy)
print ("Final SVM Accuracy: %s" % final_accuracy_score)
feature_names = zip(cv.get_feature_names(), final_model.coef_[0])
feature_to_coef = {
    word: coef for word, coef in feature_names
}
itemz = feature_to_coef.items()
list_positive = sorted(
    itemz, 
    key=lambda x: x[1], 
    reverse=True)[:5]
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular positve words")
for best_positive in list_positive:
    print (best_positive)
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular negative words")
list_negative = sorted(
    itemz, 
    key=lambda x: x[1])[:5]
for best_negative in list_negative:
    print (best_negative)
    
print('\n' * 2)
print("----------------------------------------------------------")
print(LocalTime.get(), "  Words selected report: NGram")
print("----------------------------------------------------------")
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
X = ngram_vectorizer.fit_transform(train_text)
X_test = ngram_vectorizer.transform(test_text)
best_c = Logistic_Regression.get_best_hyperparameter(X_train, y_train, y_val, X_val)
final_ngram = LogisticRegression(C=best_c)
final_ngram.fit(X, target)
final_accuracy = final_ngram.predict(X_test)
final_accuracy_score = accuracy_score(target_test, final_accuracy)
print ("Final NGram Accuracy: %s" % final_accuracy_score)
feature_names = zip(cv.get_feature_names(), final_ngram.coef_[0])
feature_to_coef = {
    word: coef for word, coef in feature_names
}
itemz = feature_to_coef.items()
list_positive = sorted(
    itemz, 
    key=lambda x: x[1], 
    reverse=True)
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular positve words")
for best_positive in list_positive[:5]:
    print (best_positive)
print("-----------------------------------------------")
print(LocalTime.get(), "--- Most popular negative words")
list_negative = sorted(
    itemz, 
    key=lambda x: x[1])
for best_negative in list_negative[:5]:
    print (best_negative)



#X_val1, y_val1 = make_blobs(n_samples=50, centers=2,
#                  random_state=0, cluster_std=0.60)
#plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, s=50, cmap='autumn')
#data = Datalook(twitter)
#data.show()

#Wordcloudz.show(twitter, 'user_description')
print('\n' * 2)
print("-----------------------------------------------")
print(LocalTime.get(), "  General word report")
print("-----------------------------------------------")
twitter['sentiment'] = twitter['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
print(LocalTime.get(), "5 random tweets with highest positive sentiment polarity: \n")
cL = twitter.loc[twitter.sentiment==1, ['full_text']].sample(5).values
for c in cL:
    print(c[0])
    print()

cL = twitter.loc[twitter.sentiment >0, ['full_text']].values
positive_sentences = Sentences1.filter1(cL)
number_positive = len(positive_sentences)
percentage_positive = "which is {0:.2f}% of all tweets".format((number_positive / len(twitter)) * 100)
print("-----------------------")
print(LocalTime.get(), "Number of positive sentiments = ", number_positive, percentage_positive)

cL = twitter.loc[twitter.sentiment < 0, ['full_text']].values
negative_sentences = Sentences1.filter1(cL)
number_negative = len(negative_sentences)
percentage_negative = "which is {0:.2f}% of all tweets".format((number_negative / len(twitter)) * 100)
print("-----------------------")
print(LocalTime.get(), "Number of negative sentiments = ", number_negative, percentage_negative)

cL = twitter.loc[twitter.sentiment==0, ['full_text']].values
neutral_sentences = Sentences1.filter1(cL)
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
