import numpy

# Sample values for A, b, c
A_matrix = numpy.array([[1, -1],
                     [2, 3]])
b = numpy.array([4, 5])
c = numpy.array([1, 2])

m, n = A_matrix.shape
identity_mat = numpy.identity(m)
objective_row = numpy.hstack((numpy.reshape([-x for x in c], (-1, 1)), numpy.zeros((m, 1))))
val_col = numpy.vstack((b.reshape(-1, 1), numpy.zeros((m+1-len(b), 1))))
print(val_col)
tableau = numpy.hstack((A_matrix, identity_mat))
objective_row = objective_row.T
tableau = numpy.vstack((tableau, objective_row.reshape(1, -1)))
tableau = numpy.hstack((tableau, val_col))
tableau = numpy.hstack((tableau, numpy.empty((m+1, 1))))
print(tableau)