import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud1 import Wordcloudz
from word_processing import Word_Processing1
from report_matricies import Report_Matricies
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

number_we_are_interested_in = 5
print('\n' * 10)
print("="*80)
print("Local current time started :", LocalTime.get())
print("="*80)
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
full_text_column = "full_text"
train, test = train_test_split(twitter1, test_size=0.2)
train_text = Sentences1.filter(train[full_text_column])
test_text = Sentences1.filter(test[full_text_column])
target = np.where(train.sentiment > 0, 1, 0)
target_test = np.where(test.sentiment > 0, 1, 0)

print("-"*100)
print("Train and test data divided")
print("Train data = ", train.shape)
print("Test data = ", test.shape)
print("-"*100)
######## Vectorise the train and test datasets
cv = CountVectorizer(binary=True)
X = cv.fit_transform(train_text)
X_test = cv.transform(test_text)
######## Build the classifiers
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)
print('\n' * 2)
print("-"*100)
print(LocalTime.get(), "  Words selected report")
print("-"*100)
best_c = Logistic_Regression.get_best_hyperparameter(X_train, y_train, y_val, X_val)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
final_accuracy = final_model.predict(X_test)
final_accuracy_score = accuracy_score(target_test, final_accuracy)
print ("Final Accuracy: %s" % final_accuracy_score)

Report_Matricies.accuracy(target_test, final_accuracy)

feature_names = zip(cv.get_feature_names(), final_model.coef_[0])
feature_to_coef = {
    word: coef for word, coef in feature_names
}
itemz = feature_to_coef.items()
list_positive = sorted(
    itemz, 
    key=lambda x: x[1], 
    reverse=True)[:number_we_are_interested_in]
print("-"*100)
print(LocalTime.get(), "--- Most popular positve words")
for best_positive in list_positive:
    print (best_positive)
print("-"*100)
print(LocalTime.get(), "--- Most popular negative words")
list_negative = sorted(
    itemz, 
    key=lambda x: x[1])[:number_we_are_interested_in]
for best_negative in list_negative:
    print (best_negative)

print('\n' * 2)
print("-"*100)
print(LocalTime.get(), "  Words selected report: SVM ")
print("-"*100)
best_c = Linear_SVM.get_best_hyperparameter(X_train, y_train, y_val, X_val)
final_svm  = LinearSVC(C=best_c)
final_svm.fit(X, target)
final_accuracy = final_svm.predict(X_test)
final_accuracy_score = accuracy_score(target_test, final_accuracy)
print ("Final SVM Accuracy: %s" % final_accuracy_score)
Report_Matricies.accuracy(target_test, final_accuracy)
feature_names = zip(cv.get_feature_names(), final_model.coef_[0])
feature_to_coef = {
    word: coef for word, coef in feature_names
}
itemz = feature_to_coef.items()
list_positive = sorted(
    itemz, 
    key=lambda x: x[1], 
    reverse=True)[:number_we_are_interested_in]
print("-"*100)
print(LocalTime.get(), "--- Most popular positve words")
for best_positive in list_positive:
    print (best_positive)
print("-"*100)
print(LocalTime.get(), "--- Most popular negative words")
list_negative = sorted(
    itemz, 
    key=lambda x: x[1])[:number_we_are_interested_in]
for best_negative in list_negative:
    print (best_negative)

print('\n' * 2)

for no_of_words in range(2,4):
    print("-"*100)
    print(LocalTime.get(), "  Words selected report: NGram where n = ", no_of_words)
    print("-"*100)
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, no_of_words))
    X = ngram_vectorizer.fit_transform(train_text)
    X_test = ngram_vectorizer.transform(test_text)
    best_c = Logistic_Regression.get_best_hyperparameter(X_train, y_train, y_val, X_val)
    final_ngram = LogisticRegression(C=best_c)
    final_ngram.fit(X, target)
    final_accuracy = final_ngram.predict(X_test)
    final_accuracy_score = accuracy_score(target_test, final_accuracy)
    print ("Final NGram Accuracy: %s" % final_accuracy_score)
    Report_Matricies.accuracy(target_test, final_accuracy)
    feature_names = zip(cv.get_feature_names(), final_ngram.coef_[0])
    feature_to_coef = {
        word: coef for word, coef in feature_names
    }
    itemz = feature_to_coef.items()
    list_positive = sorted(
        itemz, 
        key=lambda x: x[1], 
        reverse=True)
    print("-"*100)
    print(LocalTime.get(), "--- Most popular positve words")
    for best_positive in list_positive[:number_we_are_interested_in]:
        print (best_positive)
    print("-"*100)
    print(LocalTime.get(), "--- Most popular negative words")
    list_negative = sorted(
        itemz, 
        key=lambda x: x[1])
    for best_negative in list_negative[:number_we_are_interested_in]:
        print (best_negative)

#data = Datalook(twitter)
#data.show()

#Wordcloudz.show(twitter, full_text_column)
print('\n' * 2)
print("-"*100)
print(LocalTime.get(), "  General word report")
print("-"*100)
twitter['sentiment'] = twitter[full_text_column].map(lambda text: TextBlob(text).sentiment.polarity)
print(LocalTime.get(), "5 random tweets with highest positive sentiment polarity: \n")
cL = twitter.loc[twitter.sentiment==1, [full_text_column]].sample(5).values
for c in cL:
    print(c[0])
    print()

cL = twitter.loc[twitter.sentiment >0, [full_text_column]].values
positive_sentences = Sentences1.filter1(cL)
number_positive = len(positive_sentences)
percentage_positive = "which is {0:.2f}% of all tweets".format((number_positive / len(twitter)) * 100)
print("-"*100)
print(LocalTime.get(), "Number of positive sentiments = ", number_positive, percentage_positive)

cL = twitter.loc[twitter.sentiment < 0, [full_text_column]].values
negative_sentences = Sentences1.filter1(cL)
number_negative = len(negative_sentences)
percentage_negative = "which is {0:.2f}% of all tweets".format((number_negative / len(twitter)) * 100)
print("-"*100)
print(LocalTime.get(), "Number of negative sentiments = ", number_negative, percentage_negative)

cL = twitter.loc[twitter.sentiment==0, [full_text_column]].values
neutral_sentences = Sentences1.filter1(cL)
number_neutral = len(neutral_sentences)
percentage_neutral = "which is {0:.2f}% of all tweets".format((number_neutral / len(twitter)) * 100)
print("-"*100)
print(LocalTime.get(), "Number of neutral sentiments = ", number_neutral, percentage_neutral)
print("-"*100)

count_of_tweets = len(twitter)
count_of_retweets = np.sum(twitter.retweet_count)
print("Total number of tweets = ", count_of_tweets)
print("Total number of retweets = ", count_of_retweets)
print("Average number of retweets per tweet = ", count_of_retweets / count_of_tweets)
tweets_retweeted = twitter.apply(lambda x:True if x["retweet_count"] > 0 else False, axis = 1)
count_of_tweets_retweeted = len(tweets_retweeted[tweets_retweeted == True])
print(f"% of tweets retweeted = ", ( count_of_tweets_retweeted / count_of_tweets) * 100)
print("Number of tweets retweeted = ", count_of_tweets_retweeted)
print(" Maximum number of retweets {twitter.retweet_count.max()}")
print(" Maximum number of favorites {twitter.favorite_count.max()}")
print("-"*100)
print("Most retweeted: ", twitter.loc[twitter['retweet_count']==6622.0,full_text_column].values)
print("-"*100)
print("Most favourited: ",twitter.loc[twitter['favorite_count']==15559.0,[full_text_column,'user_name','user_description']].values)
print("="*80)
print("Local current time completed :", LocalTime.get())
print("="*80)
