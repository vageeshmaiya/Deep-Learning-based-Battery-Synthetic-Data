# -*- coding: utf-8 -*-
The following code has been executed in google colab
"""

pip install darts

from os.path import dirname, basename
from os import getcwd
import sys


def fix_pythonpath_if_working_locally():
    """Add the parent path to pythonpath if current working dir is darts/examples"""
    cwd = getcwd()
    if basename(cwd) == 'examples':
        sys.path.insert(0, dirname(cwd))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/MyDrive/dataset4.csv"
df = pd.read_csv(path)

df.head()

df.dropna(inplace=True)

df['Date_Time'] = pd.to_datetime(df['Date_Time'])

df.head()

ser = TimeSeries.from_dataframe(df=df,time_col="Date_Time")

scaler_en = Scaler()
series_en_transformed = scaler_en.fit_transform(ser['Voltage(V)'])
train, val = series_en_transformed.split_after(pd.Timestamp('1/30/2018  10:26:10 AM'))
#series_en_transformed.plot()
scaler_en.inverse_transform(series_en_transformed).plot()

model_nbeats = NBEATSModel(
    input_chunk_length=300,
    output_chunk_length=28,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=500,
    nr_epochs_val_period=1,
    batch_size=16,
    model_name='nbeats_run',
    force_reset=True
)

model_nbeats.fit(train, val_series=val, verbose=True)

from darts.metrics import mae

def eval_model(model):
    pred_series = model.predict(n=1000)
    plt.figure(figsize=(8,5))
    series_en_transformed.plot(label='actual')
    pred_series.plot(label='forecast')
    plt.title('MAE: {}'.format(mae(pred_series, val)))
    plt.legend()
    
eval_model(model_nbeats)

pred_series = model_nbeats.historical_forecasts(
    train,
    start=pd.Timestamp('1/29/2018  1:04:10 AM'), 
    forecast_horizon=850,
    stride=5,
    retrain=False,
    verbose=True
)

plt.figure(figsize=(7,5),edgecolor="black")
ax = plt.axes()
ser[750:1250].plot(label='Ground Truth')
scaler_en.inverse_transform(pred_series)[0:50].plot(label=('Prediction (horizon 30)'))
plt.xlabel('Time (minutes)')
plt.ylabel('Charge Capacity (Ah)')
plt.yticks(np.arange(-1,4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.savefig("/content/drive/MyDrive/ChargeCapacity_dataset4_horizon30_nbeats.pdf",bbox_inches="tight")
