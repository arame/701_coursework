import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  

class Datalook:
    def __init__(self, twitter):
        self.twitter = twitter

    def show(self):
        #lets check for null values
        self.twitter.isnull().mean()*100 
        print("self.twitter: ")
        print(self.twitter.head())
        print(f" Data Available since {self.twitter.created_at.min()}")
        print(f" Data Available upto {self.twitter.created_at.max()}")
        #lets check latest and oldest self.twitter members in the dataframe
        print(f" Data Available since {self.twitter.user_created_at.min()}")
        print(f" Data Available upto {self.twitter.user_created_at.max()}")
        print('The oldest user in the data was',self.twitter.loc[self.twitter['user_created_at'] == '2006-03-21 21:04:12', 'user_name'].values)
        print('The newest user in the data was',self.twitter.loc[self.twitter['user_created_at'] == '2019-05-19 10:49:59', 'user_name'].values)
        #lets explore created_at column
        self.twitter['created_at'] =  pd.to_datetime(self.twitter['created_at'])
        cnt_srs = self.twitter['created_at'].dt.date.value_counts()
        cnt_srs = cnt_srs.sort_index()
        plt.figure(figsize=(14,6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
        plt.xticks(rotation='vertical')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of tweets', fontsize=12)
        plt.title("Number of tweets according to dates")
        plt.show()
        #lets explore user_created_at column
        count_  = self.twitter['user_created_at'].dt.date.value_counts()
        count_ = count_[:10,]
        plt.figure(figsize=(10,5))
        sns.barplot(count_.index, count_.values, alpha=0.8)
        plt.title('Most accounts created according to date')
        plt.xticks(rotation='vertical')
        plt.ylabel('Number of accounts', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.show()
        #lets derive some columns from date colums
        self.twitter['tweeted_day_of_week'] = self.twitter['created_at'].dt.weekday_name
        self.twitter['created_day_of_week'] = self.twitter['user_created_at'].dt.weekday_name
        cnt_ = self.twitter['tweeted_day_of_week'].value_counts()
        cnt_ = cnt_.sort_index() 
        fig = {
        "data": [
            {
            "values": cnt_.values,
            "labels": cnt_.index,
            "domain": {"x": [0, .5]},
            "name": "Number of tweets per day",
            "hoverinfo":"label+percent+name",
            "hole": .3,
            "type": "pie"
            },],
        "layout": {
                "title":"Percentage of tweets per days of the week",
                "annotations": [
                    { "font": { "size": 20},
                    "showarrow": False,
                    "text": "Percentage of Tweets according to days of the week",
                        "x": 0.50,
                        "y": 1
                    },
                ]
            }
        }
        #iplot(fig)
        cnt_