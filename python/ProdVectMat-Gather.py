#!usr/bin/python3


from mpi4py import MPI
import numpy as np
import time
import sys

# Parallel vector-matrix multiplication using MPI

# Initializzing Parallel part of the code
comm = MPI.COMM_WORLD # global communicator
rank = comm.Get_rank() # extracting task id
size = comm.Get_size() # extracting number of processes

# starting time measuring
t0 = time.time()

# Read the size of the matrix from a file
with open('input_n.txt') as file:
    n = int(file.read().strip()) # removing blank characters


# and initialize the vector V, values from 1 to n
V = np.array(np.arange(1,n+1), dtype='f8')

# defining number of columns to assing to the current task (to be adjusted later)
ncolumn = n//size
remainder = n%size

# initializing the arrays for GatherV communication
recvcounts=[] # size of the block of each task
displs = []   # starting index of each block

disp=0
# computing recvcounts and displs arrays
for irank in range(size):
    displs.append(disp)
    if irank < remainder:
        incol = ncolumn + 1
    else:
        incol = ncolumn
    recvcounts.append(incol)
    disp += incol

# adjusting size of each block of the matrix
if rank < remainder:
    ncolumn += 1


# initialization of portion of matrix M of ncolumns columns for each task
# (elements' value of the whole matrix ranging from 1 to n for each row)
M = np.array([np.arange( displs[rank] +1, 
                           displs[rank] +1 + recvcounts[rank]) 
                           for i in range(n)], dtype='f8') 


# computing the multiplication on the task's portion of matrix
C = V@M


## MPI communication for retrieving the resulting vector
## using GatherV function

buffer = np.zeros(n) # receive buffer
# gather to task 0 all the local C result vectors
comm.Gatherv(C, recvbuf=(buffer, np.array(recvcounts), np.array(displs), MPI.DOUBLE), root=0)

## saving the results (first 100 elements to a file)

name = sys.argv[0].split('/')[-1].split('.')[0] # defining output file
if rank==0:
    ## output filename= "<program name>_py_output.txt"
    with open(name + '_py_output.txt', 'w') as f:
        np.savetxt(f, buffer[:100], fmt='%.6f')
    #print(buffer[495:505]) # prints for consistency check of the results
    #print(buffer[-5:])

t_tot = time.time() - t0

## saving computation time to a file for computing performance metrics
if rank == 0:
    ## filename= "times_< matrix size >_<program name>_py.txt"
    timefile = f'times_{n}_' + name + '_py.txt' 
    with open(timefile, 'a') as f:
        f.write(f'{size} {t_tot}\n')

