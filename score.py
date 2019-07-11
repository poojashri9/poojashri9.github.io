

import numpy as np
import pandas as pd
data=pd.read_csv('headbrain.csv')
x=data.iloc[:,2].values     
y=data.iloc[:,3].values
x1=data.iloc[:,2:3].values
xmean=np.mean(x)               #xm,ym=x.mean(),y.mean()
ymean=np.mean(y)
from sklearn.linear_model import LinearRegression   #it is a class
regressor=LinearRegression()
regressor.fit(x1,y)
y1=regressor.predict(x[i])     
sst=0
ssr=0
for i in range(0,len(x)):
    sst=sst+(y[i]-ymean)**2
    ssr=ssr+(y[i]-y1)
    

x1=data.iloc[:,2:3].values
    #fit is used to train the machine whose x will be in array and y as it is
m=regressor.coef_
c=regressor.intercept_
print(m)
print(c)










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








import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,0:4].values
y=data.iloc[:,4].values 


from sklearn.preprocessing import LabelEncoder
lEncoder=LabelEncoder()

x[:,3]=lEncoder.fit_transform(x[:,3])
from sklearn.preprocessing import OneHotEncoder
ohEncoder=OneHotEncoder(categorical_features=[3])

x=ohEncoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)**(1/2)
score=regressor.score(x_test,y_test)
