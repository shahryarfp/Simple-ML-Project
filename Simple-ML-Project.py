#!/usr/bin/env python
# coding: utf-8

# # CA-4
# ## Phase 0

# In[43]:


import pandas as pd
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


# In[2]:


dataset = pd.read_csv('dataset.csv')
dataset


# In[3]:


target_list = []
for mg in np.array(dataset.music_genre):
    if mg not in target_list:
        target_list.append(mg)
print('Music Genres:', target_list)


# ### Part 1:

# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# ### Part 2:

# In[6]:


pd.isna(dataset).sum()


# ### Part 3:
# #### Non-Numerical:

# In[7]:


#categorical to numeric
dataset.artist_name = dataset.artist_name.astype("category").cat.codes
dataset.track_name = dataset.track_name.astype("category").cat.codes
dataset.key = dataset.key.astype("category").cat.codes
dataset['mode'] = dataset['mode'].astype("category").cat.codes

music_genre = {"Jazz":0, "Alternative":1, "Country":2, "Rap":3, "Rock":4, "Classical":5}
dataset["music_genre"] = dataset["music_genre"].map(music_genre)


categorical_dataset = copy.deepcopy(dataset.iloc[:, [0,1,8,11,15]])
lst =categorical_dataset.columns.values
for title in lst:
    categorical_dataset[title].plot.hist()
    plt.title(title)
    plt.show()


# #### Filling Missing Values

# In[8]:


# # First Try
# dataset['duration_ms'].fillna((dataset['duration_ms'].mean()), inplace=True)
# dataset['tempo'].fillna((dataset['tempo'].mean()), inplace=True)
# dataset['artist_name'].fillna((dataset['artist_name'].mode()), inplace=True)
# pd.isna(dataset).sum()

# Second Try
def replace_with_mode_mean(x_, column):
    x = copy.deepcopy(np.array(x_))
    for i in range(len(x)):
        if pd.isnull(x[i]):
            missed_data_genre = dataset.music_genre[i]
            dataset_temp = dataset[(dataset["music_genre"] == missed_data_genre)]
            if column == 'artist_name':
                x[i] = dataset_temp[column].mode()
            else:
                x[i] = dataset_temp[column].mean()
    return x
    
x = replace_with_mode_mean(dataset.artist_name, 'artist_name')
df = pd.DataFrame(x, columns = ['artist_name'])
dataset['artist_name'] = df['artist_name']

x = replace_with_mode_mean(dataset.duration_ms, 'duration_ms')
df = pd.DataFrame(x, columns = ['duration_ms'])
dataset['duration_ms'] = df['duration_ms']

x = replace_with_mode_mean(dataset.tempo, 'tempo')
df = pd.DataFrame(x, columns = ['tempo'])
dataset['tempo'] = df['tempo']

pd.isna(dataset).sum()


# #### Numerical:

# In[9]:


numeric_dataset = copy.deepcopy(dataset.select_dtypes(include=np.number))
lst = numeric_dataset.columns.values
for title in lst:
    numeric_dataset[title].plot.hist()
    plt.title(title)
    plt.show()


# ## Phase 1
# ### Part 2:

# In[10]:


# #filling missing values ==> done before ==> first and second try
pd.isna(dataset).sum()


# In[11]:


#normalizing datas
y = dataset.music_genre
dataset.drop("music_genre", axis=1, inplace=True)
dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())
dataset['music_genre'] = y


# ### Part 7:
# #### Manual

# In[12]:


# def categorize_numerical_data(numerical_list, parts_num):
#     x = copy.deepcopy(numerical_list)
#     part_len = max(x)/parts_num
#     for i in range(parts_num):
#         for j in range(len(x)):
#             if x[j] >= parts_num*part_len:
#                 x[j] = parts_num-1
#             if x[j] >= i*part_len and x[j] < (i+1)*part_len:
#                 x[j] = i
#     return x

# def calculate_entropy(x):
#     class_list = []
#     class_types = []
#     for element in x:
#         if element not in class_types:
#             class_list.append([element, [0,0,0,0,0,0], 0])
#             class_types.append(element)
#     for class_ in class_list:
#         for i in range(len(x)):
#             if x[i] == class_[0]:
#                 class_[1][dataset.music_genre[i]] += 1
#     for class_ in class_list:
#         entropy = 0
#         for genre in class_[1]:
#             if genre == 0:
#                 continue
#             entropy += genre/sum(class_[1])*math.log2(sum(class_[1])/genre)
#         class_[2] = entropy
#     entropy = 0
#     for class_ in class_list:
#         entropy += sum(class_[1])/sum(x)*class_[2]
#     return entropy

# def calculate_info_gain(x):
#     genres = [0,0,0,0,0,0]
#     for i in range(len(x)):
#         genres[dataset.music_genre[i]] += 1
#     entropy_before = 0
#     for genre in genres:
#         entropy_before += genre/sum(genres)*math.log2(sum(genres)/genre)
#     return entropy_before - calculate_entropy(x)

# info_gain_dict = {}


# In[13]:


# categorical_dataset = copy.deepcopy(dataset.iloc[:, [0,1,8,11]])
# lst =categorical_dataset.columns.values
# for title in lst:
#     x = np.array(categorical_dataset[title])
#     info_gain_dict[title] = calculate_info_gain(x)


# In[14]:


# lst = numeric_dataset.columns.values
# for title in lst:
#     x = np.array(categorize_numerical_data(numeric_dataset[title], 30))
#     info_gain_dict[title] = calculate_info_gain(x)


# In[15]:


# print('Information Gain:')
# info_gain_dict


# #### With mutual_info_classif

# In[16]:


info_gain_dict = {}
lst = dataset.iloc[:,:15].columns.values
for title in lst:
    info_gain = mutual_info_classif(dataset[title].to_numpy().reshape(-1, 1), dataset['music_genre'].to_numpy())
    info_gain_dict[title] = float(info_gain)
info_gain_dict


# In[17]:


#sorting dictionary
info_gain_dict = dict(sorted(info_gain_dict.items(), key=lambda item: item[1]))

y_axis = list(info_gain_dict.keys())
x_axis = list(info_gain_dict.values())
plt.barh(y_axis,x_axis)
plt.ylabel('feature')
plt.xlabel('information gain')
plt.show()


# ## Phase 2
# ### Part 2:

# In[18]:


y = dataset.music_genre
# #first try
# dataset.drop("key", axis=1, inplace=True)
# dataset.drop("liveness", axis=1, inplace=True)
# dataset.drop("music_genre", axis=1, inplace=True)
# dataset.drop("mode", axis=1, inplace=True)
# dataset.drop("tempo", axis=1, inplace=True)
# dataset.drop("duration_ms", axis=1, inplace=True)
# dataset.drop("valence", axis=1, inplace=True)

#second try
dataset.drop("key", axis=1, inplace=True)
dataset.drop("liveness", axis=1, inplace=True)
dataset.drop("music_genre", axis=1, inplace=True)
dataset.drop("mode", axis=1, inplace=True)
dataset.drop("valence", axis=1, inplace=True)

x = copy.deepcopy(dataset)

x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), stratify = np.array(y), test_size=0.2)


# ### Part 3:
# ### K-Nearest-Neighbors

# In[19]:


scores_test = {}
scores_train = {}
for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    scores_test[k] = metrics.accuracy_score(y_test, y_pred_test)
    scores_train[k] = metrics.accuracy_score(y_train, y_pred_train)


# In[20]:


x_axis = list(scores_test.keys())
y_axis_test = list(scores_test.values())
y_axis_train = list(scores_train.values())
plt.plot(x_axis, y_axis_test)
plt.plot(x_axis, y_axis_train)
plt.ylabel('Accuracy')
plt.xlabel('N Neighbors')
plt.legend(('test','train'))
plt.show()

max_value_test = max(scores_test, key = scores_test.get)
print("Best N Neighbors for test data:", max_value_test)
print("Best Accuracy for test data:", scores_test[max_value_test]*100, '%')


# ### Part 4:
# ### Decision Tree

# In[21]:


max_depth_list = []
min_samples_leaf_list = []
dtree_test_accuracy_list = []
dtree_train_accuracy_list = []
for max_depth in range(1,20):
    for min_samples_leaf in range(1,15):
        dtree = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        dtree = dtree.fit(x_train, y_train)
        y_pred_test = dtree.predict(x_test)
        y_pred_train = dtree.predict(x_train)
        
        max_depth_list.append(max_depth)
        min_samples_leaf_list.append(min_samples_leaf)
        dtree_test_accuracy_list.append(metrics.accuracy_score(y_test, y_pred_test))
        dtree_train_accuracy_list.append(metrics.accuracy_score(y_train, y_pred_train))


# #### test:

# In[22]:


X = np.array(max_depth_list)
Y = np.array(min_samples_leaf_list)
Z = np.array(dtree_test_accuracy_list)

value_dict = {}
for i in range(len(max_depth_list)):
    value_dict[(X[i], Y[i])] = Z[i]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(X, Y)
Z = np.empty(X.shape)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i][j] = value_dict[(X[i][j], Y[i][j])]
        
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Max Depth')
ax.set_ylabel('Min Samples Leaf')
ax.set_zlabel('Accuracy')
plt.show()
print("Best Accuracy for test data:", np.max(Z)*100, '%')


# #### train:

# In[23]:


X = np.array(max_depth_list)
Y = np.array(min_samples_leaf_list)
Z = np.array(dtree_train_accuracy_list)
for i in range(len(max_depth_list)):
    value_dict[(X[i], Y[i])] = Z[i]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(X, Y)
Z = np.empty(X.shape)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i][j] = value_dict[(X[i][j], Y[i][j])]
        
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Max Depth')
ax.set_ylabel('Min Samples Leaf')
ax.set_zlabel('Accuracy')
plt.show()
print("Best Accuracy for train data:", np.max(Z)*100, '%')


# ### Part 6:
# #### K-Nearest-Neighbors

# In[24]:


knn = KNeighborsClassifier(n_neighbors = 41)
knn.fit(x_train, y_train)
y_pred_test = knn.predict(x_test)
print('KNN:')
print(metrics.classification_report(y_test, y_pred_test, digits=3))


# #### Decision Tree

# In[25]:


dtree = tree.DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 5)
dtree = dtree.fit(x_train, y_train)
y_pred_test = dtree.predict(x_test)
print('D-Tree:')
print(metrics.classification_report(y_test, y_pred_test, digits=3))


# ## Phase 3
# ### Part 1:

# In[33]:


# Random Forest
# for testing model
RF = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_leaf = 10)
RF = RF.fit(x_train, y_train)
y_pred_test = RF.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_test)*100, '%')


# ### Part 2:

# In[37]:


RF_results = {}
max_depth = 10
for n_estimators in range(90,110):
    for min_samples_leaf in range(7,13):
        RF = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        RF = RF.fit(x_train, y_train)
        y_pred_test = RF.predict(x_test)
        print('(',n_estimators, max_depth, min_samples_leaf,')', '==>', metrics.accuracy_score(y_test, y_pred_test))
        RF_results[(n_estimators, max_depth, min_samples_leaf)] = metrics.accuracy_score(y_test, y_pred_test)


# In[38]:


keys = list(RF_results.keys())
RF_accuracy_list = list(RF_results.values())
# n = np.array(RF_accuracy_list)
best_key = keys[np.argmax(np.array(RF_accuracy_list))]
print('best key:', best_key)
print('best accuracy:', RF_results[best_key]*100, '%')


# ### Part 3:

# In[40]:


RF = RandomForestClassifier(n_estimators = 103, max_depth = 10, min_samples_leaf = 7)
RF = RF.fit(x_train, y_train)
y_pred_test = RF.predict(x_test)
print('Random Forest:')
print(metrics.classification_report(y_test, y_pred_test, digits=3))


# ### Part 4:

# In[45]:


matrix_confusion = metrics.confusion_matrix(y_test, y_pred_test)
mc = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
mc.set_title('Confusion Matrix\n');
mc.set_xlabel('\nPredicted Values')
mc.set_ylabel('Actual Values ');
plt.show()


# In[ ]:




