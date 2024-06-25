import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import math

df = pd.read_csv("e7-htr-currernt.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'],dayfirst=True)

row_index, col_index = (df == '11/4/2019 00:00').stack().idxmax()
print(row_index)     