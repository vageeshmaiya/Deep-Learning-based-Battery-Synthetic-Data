#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install darts


# In[2]:


from os.path import dirname, basename
from os import getcwd
import sys


def fix_pythonpath_if_working_locally():
    """Add the parent path to pythonpath if current working dir is darts/examples"""
    cwd = getcwd()
    if basename(cwd) == 'examples':
        sys.path.insert(0, dirname(cwd))


# In[3]:


import pandas as pd

from darts.models import TCNModel
import darts.utils.timeseries_generation as tg
from darts.utils.likelihood_models import GaussianLikelihood
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("C://Users//Admin//Desktop/dataset4.csv") #Reading csv file


# In[5]:


df['Date_Time'] = pd.to_datetime(df['Date_Time']) #converting the dataframe file to date time format


# In[6]:


df.info()


# In[7]:


ser = TimeSeries.from_dataframe(df=df,time_col="Date_Time") # converting the dataframe to time series data


# In[8]:


scaler_en = Scaler() #scaling
series_en_transformed = scaler_en.fit_transform(ser)
train_en_transformed, val_en_transformed = series_en_transformed.split_after(pd.Timestamp('1/30/2018  10:26:10 AM'))
plt.figure(figsize=(10, 3))
series_en_transformed.plot()


# In[9]:


# model selection and initialization of model parameters
deeptcn = TCNModel(
    dropout=0,
    batch_size=16,
    n_epochs=500,
    optimizer_kwargs={'lr': 1e-3}, 
    random_state=42,
    input_chunk_length=300,
    output_chunk_length=12,
    kernel_size=3,
    num_filters=4,
    likelihood=GaussianLikelihood())

deeptcn.fit(series=train_en_transformed, verbose=True)


# In[109]:


from darts.metrics import mae


# In[110]:


# Prediction and Plotting
def eval_model(model):
    pred_series = model.predict(n=2000)
    plt.figure(figsize=(8,5))
    plt.xlabel('Time')
    plt.ylabel("Charge Capacity('Ah')")
    series_en_transformed.plot(label='actual')
    pred_series.plot(label='forecast')
    plt.title('MAE: {}'.format(mae(pred_series, val_en_transformed)))
    plt.savefig("darts_deeptcn_6_cycles.pdf",bbox_inches="tight")
    plt.legend()
    
    
eval_model(deeptcn)


# In[111]:


# Backtesting
backtest_en = deeptcn.historical_forecasts(
    series=series_en_transformed,
    num_samples=50,
    start=0.7,
    forecast_horizon=5,
    stride=5,
    retrain=False,
    verbose=True)


# In[112]:


mae(series_en_transformed, backtest_en)


# In[113]:


import numpy as np


# In[119]:


# Backtesting and Plotting
plt.figure(figsize=(7,5),edgecolor="black")
ax = plt.axes()
scaler_en.inverse_transform(series_en_transformed)[2695:3190].plot(label='actual')
scaler_en.inverse_transform(backtest_en)[0:60].plot(
    label='backtest (horizon=5)', 
    low_quantile=0.01, 
    high_quantile=0.99)
plt.legend();
plt.xlabel('Time (minutes)')
plt.ylabel('Voltage (V)')
plt.yticks(np.arange(-0.5,4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
#plt.savefig("ChargeCapacity_Dataset4_horizon5_deepTCN.pdf",bbox_inches="tight")


# In[16]:


# Mae values for Voltage and Charge Capacity obtained in backtesting
horizon = [10,30,50,80,100,200,300,400,500,600,700,800,900,1000]
mae_voltage = [0.03300610430736358,0.040354819261249765,0.045392651863248125,0.055605198909112356,0.06144693328235384,0.08786146566232034,0.11741502293934725,0.16236704465543755,0.1863151008859987,0.21096878454408405,0.24203894657461486,0.2657503843443756,0.28393446129061506,0.3218624875454618]
mae_chargecapacity = [0.027861769864558072,0.029298309573684456,0.029334995674947188,0.02938976354178376,0.041822742772139626,0.05828229633404029,0.06588024810703844,0.07160108807626284,0.07633627964995077,0.08148715309189944,0.08113592440154707,0.08935655658901619,0.10657968729560184,0.12860550478967625]


# In[17]:


import matplotlib.pyplot as plt


# In[20]:


# Plotting of mae values
fig,ax = plt.subplots()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.plot(horizon,mae_voltage,marker = "o",label = "Voltage (V)")
ax.plot(horizon,mae_chargecapacity,marker = "X",label = "Charge Capacity (Ah)")
ax.set_xlabel('Horizon')
ax.set_ylabel('MAE')
#increase trace thickness
plt.legend()
#plt.savefig("darts_deeptcn_mae_vs_horizon_dataset4.pdf",bbox_inches="tight")


# In[ ]:




