import plotly.graph_objects as go
import pandas as pd

# Sample data
data = {
    'x': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'y': [i**2 for i in range(100)]
}

df = pd.DataFrame(data)

# Create a line chart
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name='Line'))

# Update layout to include range slider
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# Show the figure
fig.show()
# TODO: Trying to plot the days with almost ideal data
start_index = 288*111 #112th day
end_index = 288*112

df_range = df.iloc[start_index:end_index]
xval = df_range.index
yval = df_range['HT R Phase Current']

fig = go.Figure()
fig.add_trace(go.Scatter(x=xval, y=yval, mode='lines+markers', name='HT R Phase Current'))
fig.update_layout(
    title='Best Day',
    xaxis_title='Index',
    yaxis_title='HT R Phase Current'
)
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)
fig.show()
# END TODO

"""
# TODO: To find out mean current at every point of day. "Ideal Day Creation"
mean_time=[]
for j in range(288):
    temp=[]
    for i in range(280):
        temp.append(df.iloc[i*288+j,df.columns.get_loc('HT R Phase Current')])
    mean_time.append(np.mean(temp))
plt.plot(range(288),mean_time)
plt.show()
# END TODO

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

plt.plot(root_sq)
plt.show()
# END TODO
""" 
