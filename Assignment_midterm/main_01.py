import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import math
import statsmodels.api as sm

df = pd.read_csv("e7-htr-currernt.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'],dayfirst=True)

# TODO : To plot the data of 10 days a time in the plot to manually pick good days.
sdate = '2019-01-01'
edate = '2019-01-10'
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['HT R Phase Current'], mode='lines', name='Line'))

fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='date',
        range=[sdate, edate]
    )
)

#fig.show()
# END TODO

# TODO: To create an ideal good day using the picked good days
good_days=['29-12-2018 00:00', '30-12-2018 00:00', '31-12-2018 00:00', '01-01-2019 00:00','29-01-2019 00:00',  '05-02-2019 00:00','06-02-2019 00:00','14-02-2019 00:00','15-02-2019 00:00','04-04-2019 00:00', '14-04-2019 00:00', '15-04-2019 00:00', '16-04-2019 00:00', '23-04-2019 00:00','29-04-2019 00:00','23-05-2019 00:00','05-06-2019 00:00']
good1=[]
for x in good_days:
    row_index, col_index = (df == x).stack().idxmax()
    good1.append(row_index)
print(good1)

good2=[]
for x in good1:
    for i in range(288):
        good2.append(int(x)+i)

good3=[]
for x in good2:
    good3.append(df.iloc[x,df.columns.get_loc('HT R Phase Current')])

mean_time=[]
for i in range(288):
    temp=[]
    for x in good1:
        temp.append(df.iloc[x+i,df.columns.get_loc('HT R Phase Current')])
    mean_time.append(np.mean(temp))
index=[x for x in range(288)]
plt.scatter(index,mean_time)
#plt.show()

# END TODO  


def mapper(input_data):
    ans = (3-2)/(510-222)*(input_data)+1
    return round(ans)

# TODO: Root sum squares
root_sq=[]
for i in range(280):
    temp=[]
    for j in range(288):
        temp.append(df.iloc[i*288+j,df.columns.get_loc('HT R Phase Current')])
    sum=0.0
    for x in temp:
        sum += (mean_time[j]-x)**2
    sum = math.sqrt(sum) 
    root_sq.append(sum)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(root_sq))), y=root_sq, mode='lines'))
highlight_indices = [mapper(x)-1 for x in good1]
print(highlight_indices)
highlight_values = [root_sq[i] for i in highlight_indices]
meanrsq=np.mean(highlight_values)
fig.add_shape(
    type="line",
    x0=0, x1=1,
    y0=meanrsq, y1=meanrsq,
    xref='paper',
    yref='y',
    name='mean root_sq for good days',
    line=dict(
        color="Orange",
        width=2,
        dash="dashdot"
    )
)
fig.add_trace(go.Scatter(x=highlight_indices, y=highlight_values, mode='markers',
                         marker=dict(size=10, color='red'), name='Good Days'))

fig.update_layout(title='Root Squares Plot',
                  xaxis_title='Index',
                  yaxis_title='Root Squares')

#fig.show()
# END TODO

def date_convert(df):
    # Convert timestamps to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract features
    df['minute_of_day'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# Separate good and bad data
good_data = date_convert(df[df['quality'] == 'good'])
bad_data = date_convert(df[df['quality'] == 'bad'])

# Define features and target
features = ['minute_of_day', 'day_of_year']
X_train = good_data[features]
y_train = good_data['current']

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)

# Train the MLR model using statsmodels
model = sm.OLS(y_train, X_train).fit()

# Preprocess bad data for validation
X_test = bad_data[features]
y_test = bad_data['current']

# Add a constant to the validation data
X_test = sm.add_constant(X_test)

# Predict and evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics manually
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')