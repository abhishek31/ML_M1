#Using Simple Linear Regression 
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CO2EMISSIONS']]
X=df
Y=df['CO2EMISSIONS'].ravel()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn.linear_model import LinearRegression 
reg=LinearRegression()
x=np.asanyarray(train[['ENGINESIZE']])
y=np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(x,y)
x1=np.asanyarray(test[['ENGINESIZE']])
yhat=reg.predict(x)
y1=np.asanyarray(test[['CO2EMISSIONS']])
print('coefficents:', reg.coef_ )
print("MSE: %.2f" % np.mean((yhat-y)**2) )
print("r-score: %.2f" % reg.score(x,y))


#Using Multiple Linear Regression 

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
X=df
Y=df['CO2EMISSIONS'].ravel()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn.linear_model import LinearRegression 
reg=LinearRegression()
x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y=np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(x,y)
x1=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
yhat=reg.predict(x)
y1=np.asanyarray(test[['CO2EMISSIONS']])
print('coefficents:', reg.coef_ )
print("MSE: %.2f" % np.mean((yhat-y)**2) )
print("r-score: %.2f" % reg.score(x,y))