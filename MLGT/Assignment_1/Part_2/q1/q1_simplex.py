"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(a) Firstly, implement simplex method covered in class from scratch to solve the LP

simplex reference:
https://www.youtube.com/watch?v=t0NkCDigq88
"""
import numpy 
import pulp
import pandas as pd
import argparse


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments


def simplex_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> list:
    """
    Implement LP solver using simplex method.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: list of pivot values simplex method encountered in the same order
    """
    pivot_value_list = []
    ################################################################
    # %% Student Code Start
    # Implement here
    # TODO: create the tableau using A_matrix, b and c. 
    m, n = A_matrix.shape
    identity_mat = numpy.identity(m)
    objective_row = numpy.hstack((numpy.reshape([-x for x in c], (-1, 1)), numpy.zeros((m, 1))))
    val_col = numpy.vstack((b.reshape(-1, 1), numpy.zeros((m+1-len(b), 1))))
    #print(val_col)
    tableau = numpy.hstack((A_matrix, identity_mat))
    objective_row = objective_row.T
    tableau = numpy.vstack((tableau, objective_row.reshape(1, -1)))
    tableau = numpy.hstack((tableau, val_col))
    tableau = numpy.hstack((tableau, numpy.empty((m+1, 1))))
    #print(tableau)
    # END TODO

    # TODO: Tableau iterations with row operations and collecting pivot values in each itration to pivot_value_list.
    while True:
        
        if all(x >= 0 for x in tableau[m, :-1]):
            break
        
        pivot_col_index = numpy.argmin(tableau[m,:-1])
        theta_list = []
        
        #Finding the least positive theta
        for i in range(m):
            if tableau[i,pivot_col_index] > 0:
                theta_list.append(tableau[i, -2] / tableau[i,pivot_col_index])
            else:
                theta_list.append(float('inf')) # took this case even if theta is zero or negative.
                #since we dont have any problem with negative theta, the above statement works. 
        
        pivot_row_index = numpy.argmin(theta_list)
        pivot_value_list.append(tableau[pivot_row_index,pivot_col_index])
        
        # pivotrow=pivotrow/pivotvalue.
        tableau[pivot_row_index,:]=tableau[pivot_row_index,:]/tableau[pivot_row_index,pivot_col_index]
        
        for i in range(m+1): #row operations.
            if i!=pivot_row_index:
                tableau[i,:]-=tableau[i,pivot_col_index]*tableau[pivot_row_index,:] 
    # END TODO    
            
    # %% Student Code End
    ################################################################

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # Read the inputs A, b, c and run solvers
    # There are 2 test cases provided to test your code, provide appropriate command line args to test different cases.
    matrix_A = pd.read_csv("{}/A.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    vector_c = pd.read_csv("{}/c.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()
    vector_b = pd.read_csv("{}/b.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()

    simplex_pivot_values = simplex_solver(matrix_A, vector_c, vector_b)
    for val in simplex_pivot_values:
        print(val)
#print('ok')