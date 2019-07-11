import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,0:4].values
y=data.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()
x[:,3]=lEncoder.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
onh=OneHotEncoder(categorical_features=[3])
x=onh.fit_transform(x).toarray()

x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,
                                                 random_state=0)
from sklearn.linear_model import LinearRegression
mregressor=LinearRegression()
mregressor.fit(x_train,y_train)

y_pred=mregressor.predict(x_test)


from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred)**(1/2))
r2=mregressor.score(x_test,y_test)
print("Rmse=",rmse)
print("score=",r2)