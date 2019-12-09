import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs

from logistic_regression import Logistic_Regression
from local_time import LocalTime
from report_matricies import Report_Matricies

class N_Gram:
    @staticmethod
    def calc(no_of_words, X_train, y_train, y_val, X_val, train_text, test_test, cv):
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
        coef0 = final_ngram.coef_[0]
        feature_n = cv.get_feature_names()
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
        for best_positive in list_positive[:5]:
            print (best_positive)
        print("-"*100)
        print(LocalTime.get(), "--- Most popular negative words")
        list_negative = sorted(
            itemz, 
            key=lambda x: x[1])
        for best_negative in list_negative[:5]:
            print (best_negative)