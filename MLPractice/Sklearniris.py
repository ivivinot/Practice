from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
knn=KNeighborsClassifier(n_neighbors=1)

iris_dataset=load_iris()
#iris = sns.load_dataset("iris")
#iris.head()
#print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset["DESCR"][:193]+"\n...")
#print("Target names: {}".format(iris_dataset["target_names"]))
print("Feature names: {}".format(iris_dataset["feature_names"]))
#print(iris_dataset['data'])
#print(iris_dataset['data'].shape)
#print(iris_dataset['target'])

x_train,x_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

iris_dataframe = pd.DataFrame(x_train, columns = iris_dataset.feature_names)
print(iris_dataframe)
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15),marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3)

iris_dataframe["target"] = y_train

sns.set(style="ticks")
sns.pairplot(iris_dataframe, hue= "target", markers=["o", "s","D"])
plt.show()

knn.fit(x_train, y_train)
print(knn)
x_new = np.array([[5,2.9,1,0.2]])
#print("x_new.shape: {}".format(x_new.shape))
prediction = knn.predict(x_new)
#print("Prediction: {}".format(prediction))
#print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(x_test)
print("test set predictions:\n {}".format(y_pred))
print("test set score: {:.2f}".format(np.mean(y_pred==y_test)))