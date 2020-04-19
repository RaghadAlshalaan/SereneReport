
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#For Descision Tree Visualization
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz 
import pydotplus

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
df_j3.head(5)

# used to store the cluster data
df_j3.to_csv('clustered_data.csv')
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=38, azim=135)

plt.cla()

ax.set_xlabel('Heart Rate')
ax.set_ylabel('Total Step')
ax.set_zlabel('Sleep Min')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c= labels.astype(np.float),alpha=1)

first_cluster = df_j3.loc[df_j3['Label']=="Low"]

second_cluster = df_j3.loc[df_j3['Label']=="LowA"]

third_cluster = df_j3.loc[df_j3['Label']=="Meduim"]

fourth_cluster = df_j3.loc[df_j3['Label']=="High"]

first_cluster.describe()

second_cluster.describe()

third_cluster.describe()

fourth_cluster.describe()



df_desc=pd.read_csv('clustered_data.csv',index_col=0)
df_desc.head()

X = df_desc[['Heartrate', 'TotalSteps' , 'sleepMin']].values
X[0:5]


y = df_desc["Label"]
y[0:5]


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3)

descTree = DecisionTreeClassifier(criterion="entropy",max_depth=None)
descTree


descTree.fit(X_trainset,y_trainset)
predTree = descTree.predict(X_testset)


dot_data = StringIO()
export_graphviz(descTree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=df_desc.columns[0:3],  
                class_names=df_desc.Label.unique().tolist())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())




# Step 1
df=pd.read_csv('clustered_data.csv',index_col=0)    # Load label data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3)  #Splitting of Data into train and test set
descTree = DecisionTreeClassifier(criterion="entropy",max_depth=None) # Define Model
descTree.fit(X_trainset,y_trainset) # train the model using training set
predTree = descTree.predict(X_testset)  # test the model using testing set

# Visualization
dot_data = StringIO()
export_graphviz(descTree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=df.columns[0:3],  
                class_names=df.Label.unique().tolist())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# Labeling of new data set
def trainData(new_df):
    #new_df=pd.read_csv('TestData.csv',index_col=0) # load the csv file
    X = new_df[['Heartrate', 'TotalSteps' , 'sleepMin']].values
    new_df["Label"] = descTree.predict(X)
    new_df.head(10)
    return new_df

