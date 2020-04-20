
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_j3 = pd.read_csv("finalData.csv",index_col=0)
df_j3["Activities"]=df_j3["VeryActiveMinutes"] +df_j3["FairlyActiveMinutes"]+df_j3["LightlyActiveMinutes"]
df_j3.drop(['VeryActiveMinutes', 'FairlyActiveMinutes','LightlyActiveMinutes','Calories',
            'Activities' , 'SedentaryMinutes' ], axis=1, inplace = True)
df_j3


# transforme data into standard scaler
from sklearn.preprocessing import StandardScaler
X = df_j3.values
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

#elbow method
Sum_of_squared_distances = []
K = range(1,12)
for k in K:
    km = KMeans(init = "k-means++", n_clusters = k)
    km = km.fit(Clus_dataSet)
    Sum_of_squared_distances.append(km.inertia_)


clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum)
k_means.fit(Clus_dataSet)
labels = k_means.labels_

df_j3["Label"] = labels
df_j3["Label"] = pd.Categorical(df_j3["Label"], df_j3["Label"].unique())
df_j3["Label"] = df_j3["Label"].cat.rename_categories(['Low', 'LowA', 'Meduim', 'High']) #LowA means that it's low but might be active

# used to store the cluster data
df_j3.to_csv('clustered_data.csv')

df_desc=pd.read_csv('clustered_data.csv',index_col=0)

X = df_desc[['Heartrate', 'TotalSteps' , 'sleepMin']].values
X[0:5]


y = df_desc["Label"]
y[0:5]


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3)

descTree = DecisionTreeClassifier(criterion="entropy",max_depth=None)


descTree.fit(X_trainset,y_trainset)
predTree = descTree.predict(X_testset)

# Step 1
df=pd.read_csv('clustered_data.csv',index_col=0)    # Load label data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3)  #Splitting of Data into train and test set
descTree = DecisionTreeClassifier(criterion="entropy",max_depth=None) # Define Model
descTree.fit(X_trainset,y_trainset) # train the model using training set
predTree = descTree.predict(X_testset)  # test the model using testing set


# Labeling of new data set
def trainData(new_df):
    #new_df=pd.read_csv('TestData.csv',index_col=0) # load the csv file
    X = new_df[['Heartrate', 'TotalSteps' , 'sleepMin']].values
    new_df["Label"] = descTree.predict(X)
    return new_df

