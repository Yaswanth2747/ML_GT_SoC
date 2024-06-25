import cvxopt
import pandas as pd
import numpy as np

def solve_least_squares(X, y):
    n, m = X.shape
    P = cvxopt.matrix(X.T @ X)
    q = cvxopt.matrix(-X.T @ y)
    G = cvxopt.matrix(0.0, (0, m))
    h = cvxopt.matrix(0.0, (0, 1))
    A = cvxopt.matrix(0.0, (0, m))
    b = cvxopt.matrix(0.0, (0, 1))
    
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x'])
    return w

X = pd.read_csv("X.csv", header=None, dtype=float).to_numpy()
Y = pd.read_csv("Y.csv", header=None, dtype=float).to_numpy()
w = solve_least_squares(X, Y)
print(w)
