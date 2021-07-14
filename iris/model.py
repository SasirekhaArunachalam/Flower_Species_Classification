import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g.pd.read_csv)
import matplotlib.pyplot as plt # for plotting and visualozing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#our dataset
diab=pd.read_csv('Iris (1).csv')
diab.fillna(0, inplace=True)
diab.info()
X=diab[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y=diab[['Species']]
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.3)
X_train.describe()
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
print("Accuracy:" ,accuracy_score(y_test, knn.predict(X_test)))