#!usr/bin/python3

import numpy as np



with open('matrix_n.txt') as file:
    n = int(file.read().strip())

print(n)

V = np.array(np.arange(1,n+1))


# initialization of portion of matrix M of ncolumns columns for each task
M = np.array([np.arange(1, n+1) for i in range(n)]) #1./array puoi con numpy 

print(M)

# computing the multiplication on the task portion of matrix
C = V@M

print(C)


# # M = np.empty((n,n))
# # np.random.seed(1)
# # M = np.random.rand(n,n)

# print('v:  shape=', V.shape, 'ndim:', V.ndim, 'size:', V.size)
# print('M:  shape=', M.shape, 'ndim:', M.ndim, 'size:', M.size)

# print(V[:5])
# print(M[:5,:5])

# # matrix product
# C = V@M
# print(C[-10:])
