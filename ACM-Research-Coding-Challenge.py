import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler

clusters = pd.read_csv("ClusterPlot.csv") # read csv file

scalar = StandardScaler() # Normalize the data so that the larger data does not overpower the others and skew the k value
# mean of V1, V2 is 0 and standard deviation of V1,V2is 1 
normalized_data = scalar.fit_transform(clusters)   # best representation of cluster data 

aggregate_distance = [] # initialize what will be the sum of squared distances 

for k in range(1,15): # display value of k from 1-15
   kfindmean=KMeans(k) 
   kfindmean.fit(normalized_data)
   aggregate_distance.append(kfindmean.inertia_) #calculate inertia
   
# output the plot to find optimal k
plt.plot(range(1,15), aggregate_distance)
plt.grid(True)
plt.xlabel('No of Clusters')
plt.ylabel('Inertia')
plt.title('Optimal K = 6')
plt.show()
