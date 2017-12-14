# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("Ticker Symbol", "GS", ["JPM", "GS"])
dbutils.widgets.dropdown("Year", "2016", [str(x) for x in range(1990, 2018)])
dbutils.widgets.dropdown("Train/Test Split Month", "10", [str(x) for x in range(1, 13)])
widget_symbol = str(dbutils.widgets.get("Ticker Symbol"))
widget_year = int(dbutils.widgets.get("Year"))
widget_month = int(dbutils.widgets.get("Train/Test Split Month"))
split_dt_str = str(widget_year)+'-'+str(widget_month)+'-01'
widget_symbol

# COMMAND ----------

import pandas as pd
from pandas import DataFrame
import numpy as np
from datetime import datetime

import quandl
from pyspark.sql import functions as F

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten,  Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#from sparkdl import KerasTransformer

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# COMMAND ----------

quandl.ApiConfig.api_key = "WH6t_pYm1wwY2rZYcYya"
raw_df = quandl.get("EOD/"+widget_symbol, collapse="daily", start_date="1990-01-01", end_date="2017-12-01")

# COMMAND ----------

raw_df['Date'] = raw_df.index.strftime("%Y-%m-%d")

# COMMAND ----------

dist_df = spark.createDataFrame(raw_df)
dist_df = dist_df.withColumn('Date', F.to_date(F.from_unixtime(F.unix_timestamp(dist_df.Date, 'yyyy-MM-dd'))))

# COMMAND ----------

dist_df.printSchema()

# COMMAND ----------

display(dist_df)

# COMMAND ----------

dist_df.count()

# COMMAND ----------

dist_df.write.format("parquet").mode("overwrite").saveAsTable("roy.stocks")

# COMMAND ----------

# MAGIC %md
# MAGIC # Daily close prices for 1 Year

# COMMAND ----------

df=spark.table("roy.stocks").select("Date", "Close").filter(F.year('Date')==widget_year ).orderBy('Date')
display(df)

# COMMAND ----------

pdf = df.toPandas()
pdf['Date'] = pdf['Date'].astype('datetime64[ns]')
pdf = pdf.set_index('Date')
ts = pdf['Close']

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecast using ARIMA

# COMMAND ----------

ts_week = ts.resample('W').mean()
ts_week_log = np.log(ts_week)
ts_week_log_diff = ts_week_log - ts_week_log.shift()

# COMMAND ----------

ts_week_log_diff.dropna(inplace=True)

# COMMAND ----------

model = ARIMA(ts_week_log, order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
print(results_ARIMA.summary())
residuals = DataFrame(results_ARIMA.resid)
print(residuals.describe())

# COMMAND ----------

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()

# COMMAND ----------

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_week_log.ix[0], index=ts_week_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

# COMMAND ----------

predictions_ARIMA = np.exp(predictions_ARIMA_log)

# COMMAND ----------

size = int(len(ts_week_log) - 15)
train = ts_week_log.loc[ts_week_log.index < pd.to_datetime(split_dt_str)]
test = ts_week_log.loc[ts_week_log.index >= pd.to_datetime(split_dt_str)]
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

arima_mse = mean_squared_error(np.exp(test), np.exp(predictions))

# COMMAND ----------

print('\n')
print('Test MSE: %.6f' % arima_mse)

predictions_series = pd.Series(predictions, index = test.index)

# COMMAND ----------

fig, ax = plt.subplots()
ax.set(title='ARIMA Predictions for Test', xlabel='Date', ylabel='Close')
ax.plot(ts.loc[ts.index >= pd.to_datetime(split_dt_str)], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecast using LSTM

# COMMAND ----------

for c in pdf.columns:
    pdf[c+'_ret'] = pdf[c].pct_change().fillna(0)
df = pdf

# COMMAND ----------

look_back = 12
sc = StandardScaler()
df.loc[:, 'Close'] = sc.fit_transform(df.loc[:, 'Close'])

train_df = df.loc[df.index < pd.to_datetime(split_dt_str)]
test_df = df.loc[df.index >= pd.to_datetime(split_dt_str)]

timeseries = np.asarray(df.Close)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
        timeseries = timeseries.T
X = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))
y = timeseries[look_back:]

predictors = ['Close']

model = Sequential()
model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=6, return_sequences=True))
model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=6, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer="rmsprop")
model.fit(X, 
          y, 
          epochs=1000, 
          batch_size=80, verbose=1, shuffle=False)

df['Pred'] = df.loc[df.index[0], 'Close']
for i in range(len(df.index)):
    if i <= look_back:
        continue
    a = None
    for c in predictors:
        b = df.loc[df.index[i-look_back:i], c].as_matrix()
        if a is None:
            a = b
        else:
            a = np.append(a,b)
        a = a
    y = model.predict(a.reshape(1,look_back*len(predictors),1))
    df.loc[df.index[i], 'Pred']=y[0][0]
df.to_hdf('DeepLearning.h5', 'Pred_LSTM')
df.loc[:, 'Close'] = sc.inverse_transform(df.loc[:, 'Close'])
df.loc[:, 'Pred'] = sc.inverse_transform(df.loc[:, 'Pred'])

# COMMAND ----------

test_df = df.loc[df.index >= pd.to_datetime(split_dt_str)]
lstm_mse = mean_squared_error(y_true=test_df.Close, y_pred=test_df.Pred)
print('\n')
print('Test MSE: %.6f' % lstm_mse)

# COMMAND ----------

fig, ax = plt.subplots()
ax.set(title='LSTM Predictions for Test', xlabel='Date', ylabel='Close')
ax.plot(test_df.Close, 'o', label='observed')
ax.plot(test_df.Pred, 'g', label='forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare MSE

# COMMAND ----------

fig, ax = plt.subplots()
objects = ('ARIMA', 'LSTM')
y_pos = np.arange(len(objects))
performance = [arima_mse,lstm_mse]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('MSE')
plt.title('Mean Squared Errors')
display(fig)

# COMMAND ----------

