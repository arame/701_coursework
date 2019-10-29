import numpy as np
import pandas as pd
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  

from files import Files
from local_time import LocalTime

t = LocalTime()
print("test")
print("---------------------------------------------------------")
print("Local current time started :", t.localtime)
print("---------------------------------------------------------")
inputFile = "Border_Crossing_Entry_Data.csv"
f = Files(inputFile)
data = pd.read_csv(f.file_path)
print(data.info())
data.head()
t = LocalTime()
print("---------------------------------------------------------")
print("Local current time completed :", t.localtime)
print("---------------------------------------------------------")
data.isnull().any()