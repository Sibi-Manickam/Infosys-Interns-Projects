import numpy as np
import pandas as pd
import math

import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf

energy = pd.read_csv('energy_dataset.csv')
energy.head()
energy.describe().T
energy.rename(columns={
   'total load actual':'Energy Consumption'

}, inplace=True)
print('completed')

energy['time'] = pd.to_datetime(energy['time'])
energy = energy.set_index('time')

