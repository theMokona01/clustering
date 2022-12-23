import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('synthetic_clustering_dataset.csv')


x = df['f1'].values
y = df['f2'].values
centers = np.random.randn(4, 2) 

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()
#Answer = 3 species

#####
#Species 1 (left side down)= (-2.0, 0.5) in f2 and (-2, 0) in f1
#Species 2 (right side down) = (-2.0, 0.5) in f2 and (0,2) in f1
#Species 3 (righ side top) = (0,2) in f2 and (-0.5,2) in f1

#####
from sklearn.cluster import KMeans, MeanShift
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X = df.loc[:]
X_numpy = X.to_numpy()
kmeans = KMeans(3)
kmeans.fit(X_numpy)

identified_clusters = kmeans.fit_predict(X_numpy)

data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['f1'],data_with_clusters['f2'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('F1')
plt.ylabel('F2')
plt.show()

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import matplotlib.pyplot as plt

dbscan = DBSCAN(eps = 0.28, min_samples = 20)
print(dbscan)

pred = dbscan.fit_predict(X_numpy)
anom_index = where(pred == -1)
values = X_numpy[anom_index]

plt.scatter(X_numpy[:,0],X_numpy[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()


x_model = data_with_clusters.iloc[:, 0:2]
y_model = data_with_clusters.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x_model, y_model, test_size=0.1, random_state=1) 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

tree.plot_tree(clf)
