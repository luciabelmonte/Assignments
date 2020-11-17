
# coding: utf-8

# ## Lucía Belmonte Valera
# 
# #### El siguiente ejercicio propone un dataset con información de todos los jugadores del videojuego FIFA 2019. El objetivo de este ejercicio es realizar un modelo de clustering y visualizar los resultados.

# ## ÍNDICE
# 
# 
# #### Librerías
# 
# 
# #### 1. Análisis exploratorio de los datos
#    - 1.1. Dimensiones de las bases de datos
#    - 1.2. Detección de valores nulos
#    - 1.3. Unión de las dos bases de datos
#    - 1.4. Cambio de valores para características categóricas
#    - 1.5. Análisis de la dispersión de los datos
#    - 1.6. Eliminar algunas de las columnas
# 
# 
# #### 2. Selección de características
#    - 2.1. Correlaciones entre las características
#    - 2.2. Eliminación de las características altamente correlacionadas ( >= 0.7)
#    - 2.3 División de la base de datos modificada
#    - 2.4. Cálculo de la importancia de las características
#    - 2.5. Selección Univariate de las características
#    - 2.6. Selección Backward Elimination de las características
#  
#  
# #### 3. Clustering
#    - 3.1. Univariate Selection Clustering
#    - 3.2. Backward Selection Clustering
#    - 3.3. Selección Random Forest Clustering
# 
# 
# #### 4. Conclusiones

# ### Librerías

# In[1]:

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# ### 1. Análisis exploratorio de los datos

# Clustering es una técnica de análisis exploratorio y no supervisado que tiene el objetivo de mostrar las relaciones existentes entre características que no son evidentes pero que sí pueden ser importantes.
# 
# Para llevar a cabo esta tarea, primero he explorado los datos disponibles en las dos bases de datos:
# 
# - La base de datos 'numeric_data' contiene 42 características, mientras que 'categorical data' contiene 6 características.
# 
# - Ambas bases de datos se relacionan por la columna 'ID' y cuentan con información sobre 14.743 jugadores de fútbol.
# 
# - Ninguna de las dos bases de datos contiene valores nulos, por lo que no he tenido que 
# realizar ninguna tarea de limpieza ni de imputación con respecto a este tipo de valores.
# 
# 
# He unido las dos bases de datos con el objetivo de enriquecer los resultados:
# 
# - He convertido las características “Preferred Foot” y “Work Rate” en variables numéricas para que pudieran ser analizas, a la misma vez que he eliminado las características “ID”, “Name”, “Nationality” y “Club”, ya contenían una gran cantidad de valores categóricos distintos que podrían dificultar la asignación de valores numéricos.
# 
# - También he querido analizar la dispersión de los datos y he podido comprobar que algunas de las características tienen la presencia de grupos, como la variable 'International Reputation, que indica que la mayoría de los jugadores representados tiene reputación internacional', o la variable 'Salaries', que indica que la mayoría de jugadores tienen sueldos altos.
# 
# 

# In[2]:

numeric_data = pd.read_csv("numeric_data.csv", sep=",")

categorical_data = pd.read_csv("categorical_data.csv", sep=",")


# In[3]:

numeric_data.head()


# In[4]:

categorical_data.head()


# #### 1.1. Dimensiones de las bases de datos

# In[5]:

numeric_data.shape


# In[6]:

categorical_data.shape


# #### 1.2. Detección de valores nulos

# In[7]:

missing_values_numeric_data = (numeric_data.isnull().sum())

missing_values_numeric_data


# In[8]:

missing_values_categorical_data = (categorical_data.isnull().sum())

missing_values_categorical_data


# #### 1.3. Unión de las dos bases de datos

# In[9]:

full_dataset = pd.merge(numeric_data, categorical_data, on='ID', how='left')


# In[10]:

full_dataset


# In[11]:

full_dataset["Nationality"].value_counts()


# In[12]:

full_dataset["Club"].value_counts()


# In[13]:

full_dataset["Preferred Foot"].value_counts()


# In[14]:

full_dataset["Work Rate"].value_counts()


# #### 1.4. Cambio de valores para características categóricas

# In[15]:

cleanup_nums = {"Preferred Foot":     {"Right": 1, "Left": 2},
                "Work Rate": {"Medium/ Medium": 1, "High/ Medium": 2, "Medium/ High": 3, "High/ High": 4,
                                  "Medium/ Low": 5, "High/ Low": 6, "Low/ Medium": 7, "Low/ High": 8, "Low/ Low": 9 }}


# In[16]:

full_dataset.replace(cleanup_nums, inplace=True)
full_dataset.head()


# #### 1.5. Análisis de la dispersión de los datos

# In[17]:

fig = plt.figure(figsize = (15,15))
ax = fig.gca()
full_dataset.hist(ax=ax)
plt.show()


# #### 1.6. Eliminar algunas de las columnas

# In[18]:

working_dataset = full_dataset.drop(columns=['ID', 'Name', 'Nationality', 'Club'])


# In[19]:

working_dataset


# ### 2. Selección de características

# Para la selección de características, he realizado los siguientes pasos:
# 
# -	Tras el primer análisis exploratorio de la información, he procedido a calcular las correlaciones existentes entre las características, resultando en la existencia de una gran cantidad, por lo que he procedido a eliminar aquellas en las que la correlación fuera igual o mayor de 0.7.
# 
# 
# -	He dividido la base de datos resultante para poder aplicar algoritmos que me pudieran ayudar a determinar la importancia de las características restantes: Linear Regression, Logistic Regression, Random Forest Regressor, Random Forest Classifier, Univariate Selection y Backward Elimination.
# 

# #### 2.1. Correlaciones entre las características

# In[20]:

correlations = working_dataset.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()


# #### 2.2. Eliminación de las características altamente correlacionadas ( >= 0.7)

# In[21]:

columns = np.full((correlations.shape[0],), True, dtype=bool)
for i in range(correlations.shape[0]):
    for j in range(i+1, correlations.shape[0]):
        if correlations.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
selected_columns = working_dataset.columns[columns]
data = working_dataset[selected_columns]


# #### 2.3 División de la base de datos modificada

# In[22]:

data


# In[23]:

X = data.iloc[:,0:19]
y = data.iloc[:,19]


# #### 2.4. Cálculo de la importancia de las características

# In[24]:

model = LinearRegression()

model.fit(X, y)

importance = model.coef_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[25]:

model = LogisticRegression()

model.fit(X, y)

importance = model.coef_[0]

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[26]:

model = RandomForestRegressor()

model.fit(X, y)

importance = model.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[27]:

model = RandomForestClassifier()

model.fit(X, y)

importance = model.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# #### 2.5. Selección Univariate de las características

# In[28]:

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# #### 2.6. Selección Backward Elimination de las características

# In[29]:

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()
model.pvalues


# In[30]:

cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# ### 3. Clustering

# Para realizar el Clustering, he utilizado las distintas variables que he obtenido en la selección de características anterior y he escogido los resultados de los siguientes métodos para aplicarlos utilizando el algoritmo K-Means: Univariate Selection, Backward Selection y Random Forest.
# 
# -	Para seleccionar el número ideal de clusters, he utilizado la fórmula WCSS (Within-Cluster Sum of Square).
# 
# -	7 clusters en Univariate Selection.
# 
# -	12 clusters en Backward Selection.
# 
# -	7 clusters en Random Forest.
# 

# #### 3.1. Univariate Selection Clustering

# In[31]:

univariate_selection = data[['Wage', 'Interceptions', 'Finishing', 'Aggression', 'Acceleration', 'Stamina', 'HeadingAccuracy', 'Jumping', 'Weight', 'Overall']].copy()


# In[32]:

data1 = univariate_selection.values[:, 0:35]


# In[33]:

kmean=KMeans(n_clusters=3)
kmean.fit(data1)


# In[34]:

kmean.cluster_centers_


# In[35]:

kmean.labels_


# In[36]:

wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(data1)
    wcss.append(kmeans.inertia_)
    print('Cluster', i, 'Inertia', kmeans.inertia_)
    
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[37]:

# Number of clusters
k = 7
# Number of training data
n = data1.shape[0]
# Number of features in the data
c = data1.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data1, axis = 0)
std = np.std(data1, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data1[i, 0], data1[i,1], s=7)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)

plt.show()


# #### 3.2. Backward Selection Clustering

# In[38]:

backward_selection = data[['Age', 'Overall', 'Finishing', 'HeadingAccuracy', 'Acceleration', 'Jumping', 'Stamina', 'Interceptions', 'Preferred Foot']].copy()


# In[39]:

data2 = backward_selection.values[:, 0:35]


# In[40]:

kmean=KMeans(n_clusters=3)
kmean.fit(data2)


# In[41]:

kmean.cluster_centers_


# In[42]:

kmean.labels_


# In[43]:

wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)
    print('Cluster', i, 'Inertia', kmeans.inertia_)
    
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[44]:

# Number of clusters
k = 12
# Number of training data
n = data2.shape[0]
# Number of features in the data
c = data2.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data2, axis = 0)
std = np.std(data2, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data2[i, 0], data2[i,1], s=7)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)

plt.show()


# #### 3.3. Selección Random Forest Clustering

# In[45]:

random_forest = data[['Age', 'Overall', 'Potential', 'Wage', 'Weight', 'Finishing', 'HeadingAccuracy', 'Acceleration', 'Jumping', 'Stamina', 'Aggression', 'Interceptions', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Preferred Foot']].copy()


# In[46]:

data3 = random_forest.values[:, 0:35]


# In[47]:

kmean=KMeans(n_clusters=3)
kmean.fit(data3)


# In[48]:

kmean.cluster_centers_


# In[49]:

kmean.labels_


# In[50]:

wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(data3)
    wcss.append(kmeans.inertia_)
    print('Cluster', i, 'Inertia', kmeans.inertia_)
    
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') ##WCSS stands for total within-cluster sum of square
plt.show()


# In[51]:

# Number of clusters
k = 7
# Number of training data
n = data3.shape[0]
# Number of features in the data
c = data3.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data3, axis = 0)
std = np.std(data3, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data3[i, 0], data3[i,1], s=7)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)

plt.show()


# ### 4. Conclusiones

# En mi opinión, la clave para que el Clustering funcione correctamente se encuentra en la selección de las características que se incluyan en el modelo de Machine Learning, ya que con ello se evita que haya características redundantes, irrelevantes o correlacionadas entre ellas, así como mejorar la eficiencia del modelo.
# 
# En este caso, no he podido conseguir una diferenciación clara de clusters. Quizás el modelo más cercano ha podido ser el que he realizado usando Univariate Selection, pero en este caso tampoco se ve una clara diferenciación entre clusters.

# In[ ]:



