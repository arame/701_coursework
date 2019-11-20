import numpy as np
import pandas as pd
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  
from datalook import Datalook
from files import Files
from local_time import LocalTime

t = LocalTime()
print("=========================================================")
print("Local current time started :", t.localtime)
print("=========================================================")
twitter_file = "auspol2019.csv"
f1 = Files(twitter_file)
geocode_file = "location_geocode.csv"
f2 = Files(geocode_file)
twitter = pd.read_csv(f1.file_path, parse_dates=['created_at','user_created_at'])
geocodes = pd.read_csv(f2.file_path)
twitter = twitter.merge(geocodes, how='inner', left_on='user_location', right_on='name')
twitter = twitter.drop('name',axis =1)
#lets check for null values
twitter.isnull().mean()*100
data = Datalook(twitter)
data.show()
t = LocalTime()
print("=========================================================")
print("Local current time completed :", t.localtime)
print("=========================================================")
