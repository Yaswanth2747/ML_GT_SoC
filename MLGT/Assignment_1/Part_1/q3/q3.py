from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    # TODO
    return v*(k*t-(1-np.exp(-k*t)))/k
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """
    v = 0.0
    k = 0.0
    # TODO: fit a curve using SciPy to estimate v and k
    t_data =df['t'].values
    S_data =df['S'].values
    parameters, covariance = curve_fit(func, t_data, S_data)
    v, k = parameters
    #S_fit = func(t_data, v_fit, k_fit)
    # END TODO
    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = round(v,4)
    k = round(k,4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    final=[]
    
    for i in df['t'].values:
        final.append(func(i,v,k))
        
        
plt.scatter(df['t'].values, df['S'].values, label='data', color='blue',marker='*')
plt.plot(df['t'].values, final, label=f'fit: v={v}, k={k}',color='red')
plt.xlabel('time')
plt.ylabel('Displacement(S)')
plt.legend()
plt.savefig('fit_curve.png')
plt.show()
# END TODO