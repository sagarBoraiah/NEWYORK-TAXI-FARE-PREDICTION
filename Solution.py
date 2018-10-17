
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
train_df = pd.read_csv('train.csv', nrows=1000000)
train_df.shape


# In[8]:


#remove unwanted data. After multiple iterations and runs, efficient way to data clean

required_data = (train_df['fare_amount'].between(2.5, 200) & train_df['passenger_count'].between(0, 6) & 
                train_df['pickup_longitude'].between(-74.5, -72.5) & train_df['dropoff_longitude'].between(-74.5, -72.5) &
                train_df['pickup_latitude'].between(40, 42) & train_df['dropoff_latitude'].between(40, 42))


# In[9]:


train_df.shape


# In[10]:


train_df = train_df[required_data]


# In[11]:


train_df.shape


# In[16]:


train_df.head()

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
# In[18]:


train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])


# In[19]:


def extractdatetime(data):
    data['hour'] = data['pickup_datetime'].dt.hour
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['date'] = data['pickup_datetime'].dt.day
    data['day'] = data['pickup_datetime'].dt.dayofweek
    return data

train_df=extractdatetime(train_df)


# In[20]:


train_df.head()


# In[145]:


#Thanks to Stack_Overflow. Another efficient way of calculating distnce when co-or are given (Haversine Distance)
def distance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude])
    dlong = dropoff_longitude - pickup_longitude
    dlat = dropoff_latitude - pickup_latitude
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlong/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    return distance

train_df['distance'] = distance(train_df['pickup_longitude'], train_df['pickup_latitude'],
                            train_df['dropoff_longitude'], train_df['dropoff_latitude'])


# In[23]:


train_df.head(1)


# In[24]:


print(train_df.corrwith(train_df['fare_amount']))


# In[25]:


def absolute_coordinates(data):
    data['absolute_longitude'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['absolute_latitude'] = (data.dropoff_latitude - data.pickup_latitude).abs()
    return data

train_df=absolute_coordinates(train_df)


# In[26]:


train_df.head(2)


# In[27]:


def E_distance(lat1, long1, lat2, long2):
    sqlat=(lat1-lat2)**2
    sqlong=(long1-long2)**2
    e_distance = np.sqrt(sqlat+sqlong)
    return e_distance


# In[28]:


train_df['e_distance'] = E_distance(train_df.pickup_latitude, train_df.pickup_longitude, 
                               train_df.dropoff_latitude, train_df.dropoff_longitude)


# In[29]:


train_df.head(5)


# In[30]:


print(train_df.corrwith(train_df['fare_amount']))


# In[146]:


import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = train_df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(train_df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[147]:



import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = train_df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(train_df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[33]:


print(train_df.corrwith(train_df['fare_amount']))


# In[35]:


train_df.corr(method='pearson')


# In[38]:


import matplotlib.pyplot as plt

plt.scatter(train_df.distance, train_df.fare_amount)
plt.xlabel('distance mile')
plt.ylabel('fare price')

# theta here is estimated by hand
theta = (16, 4.0)
x = np.linspace(0.1, 3, 50)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


# In[39]:


import matplotlib.pyplot as plt

plt.scatter(train_df.e_distance, train_df.fare_amount)
plt.xlabel('Eucledian distance mile')
plt.ylabel('fare price')

# theta here is estimated by hand
theta = (16, 4.0)
x = np.linspace(0.1, 3, 50)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


# In[41]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['e_distance'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('eucledian distance')
plt.ylabel('Fare')


# In[42]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['distance'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('distance')
plt.ylabel('Fare')


# In[43]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['hour'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('time of a day')
plt.ylabel('Fare')


# In[44]:


import matplotlib.pyplot as plt

plt.scatter(train_df.hour, train_df.fare_amount)
plt.xlabel('Time of a day')
plt.ylabel('fare price')

# theta here is estimated by hand
theta = (16, 4.0)
x = np.linspace(0.1, 3, 50)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


# In[45]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['hour'], y=train_df['e_distance'], s=2)
plt.xlabel('time of a day')
plt.ylabel('Eucledian distance')


# In[46]:


import matplotlib.pyplot as plt

plt.scatter(train_df.hour, train_df.e_distance)
plt.xlabel('Time of a day')
plt.ylabel('eucledian distance')

# theta here is estimated by hand
theta = (16, 4.0)
x = np.linspace(0.1, 3, 50)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


# In[47]:


train_df.head(2)


# In[48]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['fare_amount'], y=train_df['absolute_longitude'], s=2)
plt.xlabel('Fare amount')
plt.ylabel('Absolute Longitude')


# In[49]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['fare_amount'], y=train_df['absolute_latitude'], s=2)
plt.xlabel('Fare amount')
plt.ylabel('Absolute Latitude')


# In[51]:


plt.figure(figsize=(15,7))
plt.hist(train_df['hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')


# In[52]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['absolute_latitude'], y=train_df['fare_amount'], s=2)
plt.xlabel('absolute latitude')
plt.ylabel('fare amount')


# In[53]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['absolute_latitude'], y=train_df['e_distance'], s=2)
plt.xlabel('absolute latitude')
plt.ylabel('e_distance')


# In[54]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['absolute_longitude'], y=train_df['e_distance'], s=2)
plt.xlabel('absolute longitude')
plt.ylabel('e_distance')


# In[55]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['day'], y=train_df['e_distance'], s=2)
plt.xlabel('Week Day')
plt.ylabel('e_distance')


# In[57]:


plt.figure(figsize=(15,7))
plt.hist(train_df['hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')


# In[59]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['day'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('Day of Week')
plt.ylabel('Fare')


# In[61]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_df['date'], y=train_df['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')


# In[65]:


plt.figure(figsize=(15,7))
plt.hist(train_df['fare_amount'], bins=100)
plt.xlabel('Fare')
plt.ylabel('Frequency')


# In[66]:


plt.figure(figsize=(15,7))
plt.hist(train_df['e_distance'], bins=100)
plt.xlabel('e_distance')
plt.ylabel('Frequency')


# In[67]:


train_df.corr(method='pearson')


# In[68]:


train_df.head(2)


# In[69]:


train_df.shape


# In[70]:


train_df_train = train_df[:600000]
train_df_test = train_df[600001:]


# In[72]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train_df_train=train_df_train.drop(['key','pickup_datetime'], axis = 1)
train_df_test=train_df_test.drop(['key','pickup_datetime'], axis = 1)


# In[73]:


train_df_train.shape


# In[74]:


train_df_test.shape


# In[75]:


train_df_train=train_df_train.drop(['key','pickup_datetime'], axis = 1)


# In[76]:


train_df_train.shape


# In[78]:


train_df_test.head(1)


# In[79]:


train_df_train = train_df[:600000]
train_df_train=train_df_train.drop(['key','pickup_datetime'], axis = 1)
train_df_train.shape


# In[80]:


train_df_test.shape


# In[81]:


train_df_test.head(1)


# In[82]:


train_df_train.head(1)


# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = train_df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(train_df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[84]:


train_df_train.corr(method='pearson')


# In[85]:


train_df_train=train_df_train.drop(['hour','passenger_count', 'month', 'date', 'day'], axis = 1)
train_df_test=train_df_test.drop(['hour','passenger_count', 'month', 'date', 'day'], axis = 1)


# In[86]:


train_df_train.head(1)


# In[87]:


train_df_test.head(1)


# In[149]:


import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = train_df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(train_df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[89]:


X = train_df_train.drop('fare_amount',axis=1)
y = y = train_df_train[['fare_amount']]


# In[91]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[92]:


from sklearn.metrics import mean_squared_error, explained_variance_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[150]:


#Thanks to Abhishek Reddy Y N for helping me with this API
from sklearn.metrics import mean_squared_error

standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)
lin_model = LinearRegression()
lin_model.fit(rescaled_X_train, y_train)
pred = lin_model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[151]:


print(pred)


# In[95]:


train_df_train = train_df


# In[96]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train_df_train=train_df_train.drop(['key','pickup_datetime'], axis = 1)


# In[97]:


train_df_train.head()


# In[152]:


import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = train_df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(train_df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[100]:


from sklearn.metrics import mean_squared_error, explained_variance_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[153]:


from sklearn.metrics import mean_squared_error

standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)
lin_model = LinearRegression()
lin_model.fit(rescaled_X_train, y_train)
pred = lin_model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[102]:


train_df_train=train_df_train.drop(['passenger_count','hour','date','month','day'], axis = 1)


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# In[154]:


from sklearn.metrics import mean_squared_error

standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)
lin_model = LinearRegression()
lin_model.fit(rescaled_X_train, y_train)
pred = lin_model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[141]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[155]:


from sklearn.metrics import mean_squared_error

standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)
lin_model = LinearRegression()
lin_model.fit(rescaled_X_train, y_train)
pred = lin_model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[156]:


print(pred)


# In[157]:


train_df_train.corr(method='pearson')


# In[158]:


train_df_train.shape


# In[159]:


train_df_train.head(1)


# In[160]:


test_df = pd.read_csv('test.csv')


# In[161]:


test_df.shape


# In[162]:


test_df.isnull().sum().sort_values(ascending=False)


# In[163]:


test_df.head()


# In[164]:


test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])


# In[165]:


test_df=extractdatetime(test_df)
test_df['distance'] = distance(test_df['pickup_longitude'], test_df['pickup_latitude'],
                            test_df['dropoff_longitude'], test_df['dropoff_latitude'])
test_df=absolute_coordinates(test_df)
test_df['e_distance'] = E_distance(test_df.pickup_latitude, test_df.pickup_longitude, 
                               test_df.dropoff_latitude, test_df.dropoff_longitude)


# In[166]:


test_df.head(2)


# In[167]:


train_df_train.head(2)


# In[168]:


test_df=test_df.drop(['key','pickup_datetime', 'hour', 'month', 'date', 'day'], axis = 1)


# In[169]:


train_df_train.head(2)


# In[170]:


test_df.head(2)


# In[173]:


train_df_train.shape


# In[172]:


test_df.shape


# In[174]:


test_df=test_df.drop(['passenger_count'], axis = 1)


# In[175]:


test_df.shape


# In[176]:


train_df_train.shape


# In[222]:


#X = train_df_train.drop('fare_amount',axis=1)
#y = y = train_df_train[['fare_amount']]
X_test = test_df
print(len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)
lin_model = LinearRegression()
lin_model.fit(rescaled_X_train, y_train)
pred = lin_model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,pred))
print(error)
print (len(pred))

standard_scaler = StandardScaler().fit(y_train)
rescaled_y_train = standard_scaler.transform(y_train)

lin_model.fit(X, y)


# In[226]:


pred = lin_model.predict(test_df)
print (len(pred))
#error = np.sqrt(mean_squared_error(test_df,pred))


# In[227]:


submission = pd.read_csv('sample_submission.csv')
submission['fare_amount'] = pred
submission.to_csv('submission_1.csv', index=False)
submission.head(20)


# In[228]:


submission.describe()


# In[229]:


import numpy as np
import pandas as pd
df = pd.read_csv('train.csv', nrows=5000000)
df.shape


# In[230]:


required_data = (df['fare_amount'].between(2.5, 200) & df['passenger_count'].between(0, 6) & 
                df['pickup_longitude'].between(-74.5, -72.5) & df['dropoff_longitude'].between(-74.5, -72.5) &
                df['pickup_latitude'].between(40, 42) & df['dropoff_latitude'].between(40, 42))


# In[231]:


df = df[required_data]


# In[232]:


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


# In[234]:


def extractdatetime(data):
    data['hour'] = data['pickup_datetime'].dt.hour
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['date'] = data['pickup_datetime'].dt.day
    data['day'] = data['pickup_datetime'].dt.dayofweek
    return data

df=extractdatetime(df)


# In[235]:


#Thanks to Stack_Overflow. Another efficient way of calculating distnce when co-or are given (Haversine Distance)
def distance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude])
    dlong = dropoff_longitude - pickup_longitude
    dlat = dropoff_latitude - pickup_latitude
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlong/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    return distance

df['distance'] = distance(df['pickup_longitude'], df['pickup_latitude'],
                            df['dropoff_longitude'], df['dropoff_latitude'])


# In[236]:


def absolute_coordinates(data):
    data['absolute_longitude'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['absolute_latitude'] = (data.dropoff_latitude - data.pickup_latitude).abs()
    return data

df=absolute_coordinates(df)


# In[237]:


def E_distance(lat1, long1, lat2, long2):
    sqlat=(lat1-lat2)**2
    sqlong=(long1-long2)**2
    e_distance = np.sqrt(sqlat+sqlong)
    return e_distance

df['e_distance'] = E_distance(df.pickup_latitude, df.pickup_longitude, 
                               df.dropoff_latitude, df.dropoff_longitude)


# In[238]:


df.shape


# In[239]:


df.head(2)


# In[240]:


print(df.corrwith(train_df['fare_amount']))


# In[241]:


import seaborn as sns
import matplotlib.pyplot as plt

spearman_correlation = df.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(10, 'fare_amount').index
correlationmap = np.corrcoef(df[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns_new.values, xticklabels=columns_new.values)

plt.show()


# In[243]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# In[252]:


X = train_df_train.drop('fare_amount',axis=1)
y = y = train_df_train[['fare_amount']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


standard_scaler = StandardScaler().fit(X_train)
rescaled_X_train = standard_scaler.transform(X_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(rescaled_X_train, y_train)
rf_predict = rf.predict(X_test)
#error = np.sqrt(mean_squared_error(y_test,pred))
print(rf_predict)


# In[246]:


train_df_train.shape


# In[247]:


len(y_test)


# In[248]:


len(X_test)


# In[249]:


len(rescaled_X_train)


# In[250]:


len(y_train)


# In[253]:


rf.fit(X, y)


# In[254]:


rf_predict = rf.predict(X_test)


# In[255]:


print (rf_predict)


# In[257]:


y_train.head(5)


# In[261]:


for i in range(10):
    print (rf_predict[i])


# In[262]:


y_train.head(10)


# In[267]:


rf_predict = rf.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,rf_predict))
print(error)


# In[264]:


print(rf_predict)


# In[268]:


test_df.shape


# In[269]:


X.shape


# In[270]:


rf_predict = rf.predict(test_df)
print (rf_predict)
print(pred)


# In[278]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
ridge_predict_test = ridge.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,ridge_predict_test))
print(error)
print(ridge_predict_test)


ridge.fit(X, y)
ridge_predict=ridge.predict(test_df)
print(ridge_predict)


# In[274]:


submission['fare_amount'] = rf_predict
submission.to_csv('submission_3.csv', index=False)
submission.head(20)


# In[279]:


submission['fare_amount'] = ridge_predict
submission.to_csv('submission_4.csv', index=False)
submission.head(20)


# In[281]:


from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.1,normalize=True, max_iter=1e5)

lassoreg.fit(X_train, y_train)
lassoreg_predict_test = ridge.predict(X_test)
error_lasso = np.sqrt(mean_squared_error(y_test,lassoreg_predict_test))
print(error_lasso)
print(lassoreg_predict_test)


ridge.fit(X, y)
lassoreg_predict=ridge.predict(test_df)
print(lassoreg_predict)


# In[282]:


submission['fare_amount'] = lassoreg_predict
submission.to_csv('submission_5.csv', index=False)
submission.head(20)


# In[283]:


train_df.head(2)


# In[1]:


from sklearn.neural_network import MLPClassifier


# In[2]:


mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

