#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install darts


# In[5]:


from os.path import dirname, basename
from os import getcwd
import sys


def fix_pythonpath_if_working_locally():
    """Add the parent path to pythonpath if current working dir is darts/examples"""
    cwd = getcwd()
    if basename(cwd) == 'examples':
        sys.path.insert(0, dirname(cwd))


# In[6]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
import darts.utils.timeseries_generation as tg
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import GaussianLikelihood

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


# In[8]:

# Reading the csv file
df = pd.read_csv("C://Users//Admin//Desktop/dataset4.csv")


# In[9]:

# Converting the pandas data frame to date time format
df['Date_Time'] = pd.to_datetime(df['Date_Time'])


# In[10]:

# Converting the data frame into time series format
ser = TimeSeries.from_dataframe(df=df,time_col="Date_Time")


# In[11]:

# Scaling
scaler_en = Scaler()
series_en_transformed = scaler_en.fit_transform(ser)
train_en_transformed, val_en_transformed = series_en_transformed.split_after(pd.Timestamp('1/30/2018  10:26:10 AM'))
series_en_transformed.plot()


# In[12]:

# Initializing the model parameters
model_en = RNNModel(
    model='LSTM',
    hidden_dim=20,
    n_rnn_layers=2,
    dropout=0.2,
    batch_size=16,
    n_epochs=500,
    optimizer_kwargs={'lr': 1e-3},
    random_state=42,
    training_length=243,
    input_chunk_length=162,
    likelihood=GaussianLikelihood()
)


# In[13]:


model_en.fit(series=train_en_transformed, 
             verbose=True)


# In[2]:


from darts.metrics import mae


# In[13]:

# Prediction and plotting
def eval_model(model):
    pred_series = model.predict(n=1000)
    plt.figure(figsize=(8,5))
    series_en_transformed.plot(label='actual')
    pred_series.plot(label='forecast')
    plt.title('MAE: {:.2f}%'.format(mae(pred_series, val_en_transformed)))
    plt.legend()
    
    plt.savefig("darts_deepAR_6_cycles.pdf",bbox_inches="tight")
    
eval_model(model_en)


# In[ ]:

# Backtesting
backtest_en = model_en.historical_forecasts(series=series_en_transformed,
                                            num_samples=50,
                                            start=0.7,
                                            forecast_horizon=500,
                                            stride=5,
                                            retrain=False,
                                            verbose=True)


# In[3]:


mae(series_en_transformed, backtest_en)


# In[22]:


plt.figure(figsize=(7,5),edgecolor="black")
ax = plt.axes()
scaler_en.inverse_transform(series_en_transformed)[2695:3190].plot(label='Ground Truth')
scaler_en.inverse_transform(backtest_en)[0:60].plot(
    label='Prediction (horizon=30)', 
    low_quantile=0.01, 
    high_quantile=0.99)
plt.legend();
plt.xlabel('Time (minutes)')
plt.ylabel('Charge Capacity (Ah)')
plt.yticks(np.arange(2.7,5.7))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.savefig("ChargeCapacity_Dataset4_horizon30_deepAR.pdf",bbox_inches="tight")


# In[14]:


horizon = [10,30,50,80,100,200,300,400,500,600,700,800,900,1000] 
mae_voltage = [0.028240742225291323,0.03547731229501051,0.04344310391205137,0.04400652467326994,0.048100459340481745,0.06253215910141811,0.07706666575102646,0.10093288643900705,0.11987181930375958,0.1368720073684441,0.1544686256187071,0.18358829242283906,0.19671005450972537,0.22195815399518584]
mae_chargecapacity = [0.01858019844450955,0.05057368232783319,0.06815149476276788,0.06167047452813972,0.09000938761613485,0.14842236463693514,0.18415400735562212,0.21761887794608506,0.23387743856649765,0.2738795186671694,0.2880329654293883,0.28689844597817876,0.30570481318795645,0.30828611646741305]


# In[16]:


fig,ax = plt.subplots()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.plot(horizon,mae_voltage,marker = "o",label = "Voltage (V)")
ax.plot(horizon,mae_chargecapacity,marker = "X",label = "Charge Capacity (Ah)")
ax.set_xlabel('Horizon')
ax.set_ylabel('MAE')
plt.legend()
plt.savefig("darts_deepar_mae_vs_horizon_dataset4.pdf",bbox_inches="tight")

