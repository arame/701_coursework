import numpy as np
import pandas as pd
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import seaborn as sns  

from files import Files
from local_time import LocalTime

t = LocalTime()

print("=========================================================")
print("Local current time started :", t.localtime)
print("=========================================================")
inputFile = "MissingMigrants-Global-2019-11-04T14-04-17.csv"
f = Files(inputFile)
migrants = pd.read_csv(f.file_path)
migrants['latitude'], migrants['longitude'] = migrants['Location Coordinates'].str.split(',',1).str
migrants['longitude'] = [float(x) for x in migrants['longitude']]
migrants['latitude'] = [float(x) for x in migrants['latitude']]

for col in list(migrants.columns):
    # Select columns that should be string
    if ('Total Dead and Missing' in col):
        # Convert the data type to float
        migrants[col] = migrants[col].str.replace(",","").astype(float)
    
    if ('Reported Date' in col):
        # Convert the data type to date
        migrants[col] = pd.to_datetime(migrants[col])

no_of_bins = migrants['Reported Year'].max() - migrants['Reported Year'].min() + 1
plt.style.use('fivethirtyeight')
plt.hist(migrants['Reported Year'].dropna(), bins = no_of_bins, edgecolor = 'k')
#plt.figure(figsize=(10, 10))
plt.ylabel('Total Dead and Missing')
plt.xlabel('Reported Year') 
plt.title('Total Dead and Missing per Year')
plt.show()
t = LocalTime()
print("---------------------------------------------------------")
print("Local current time completed :", t.localtime)
print("---------------------------------------------------------")
