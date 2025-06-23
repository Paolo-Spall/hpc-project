#!usr/bin/python3


from mpi4py import MPI
import numpy as np
import time
import sys
# Parallel vector-matrix multiplication using MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()

# Read the size of the matrix from a file
with open('input_n.txt') as file:
    n = int(file.read().strip())


# and initialize the vector V
V = np.array(np.arange(1,n+1), dtype='f8')

ncolumn = n//size
remainder = n%size

recvcounts=[]
displs = []

disp=0
for irank in range(size):
    displs.append(disp)
    if irank < remainder:
        incol = ncolumn + 1
    else:
        incol = ncolumn
    recvcounts.append(incol)
    disp += incol


if rank < remainder:
    ncolumn += 1

#aggiungere (remainder//rank)*rank

# number of column to add to keep into account of uneven distribution
if rank >= remainder:
    addition = rank
else:
    addition = 0

# initialization of portion of matrix M of ncolumns columns for each task
# (elements' value of the whole matrix ranging from 1 to n for each row)
M = np.array([np.arange( rank*ncolumn + addition +1, 
                           (rank+1)*ncolumn + addition +1) 
                           for i in range(n)], dtype='f8') #1./array puoi con numpy 


# computing the multiplication on the task portion of matrix
C = V@M


# buffer = np.zeros(n)
# comm.Gatherv(C, recvbuf=(buffer, np.array(recvcounts), np.array(displs), MPI.DOUBLE), root=0)


name = sys.argv[0].split('/')[-1].split('.')[0] 
output_file = f'{name}_py_output.txt'


if rank==0:
    n_print = min(100, n) 
    with open(name + '_py_output.txt', 'w') as f:
        np.savetxt(f, C[:n_print], fmt='%.6f')
        n_print -= len(C[:n_print])
        print(C[-5:])

        for itask in range(1, size):
            buffer = np.zeros(recvcounts[itask])
            comm.Recv(buffer, source = itask)
            if n_print > 0:
                np.savetxt(f, buffer[:n_print], fmt='%.6f')
                n_print -= len(buffer[:n_print])
            if itask == 1:
                print(buffer[:5])
else:
    comm.Send(C, dest=0)


t_tot = time.time() - t0

if rank == 0:
    timefile = f'times_{n}_' + name + '_py.txt' 
    #print('Execution time saved in:',timefile)
    with open(timefile, 'a') as f:
        f.write(f'{size} {t_tot}\n')

