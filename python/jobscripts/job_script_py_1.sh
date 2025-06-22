#!/bin/bash -x
#
#PBS -N DAE ! job name
#PBS -j oe !merge std-err and std-out
#PBS -q s3par ! [cite: 2] queue
#PBS -l nodes=1:ppn=16 # max 1 task, max 1 node
#PBS -l walltime=00:10:00

module purge
module load oneapi/compiler
module load oneapi/mkl
module load oneapi/mpi
module load IMPI/west5.4.0

MYHOME="/home/corsohpc7/hpc-project-03"
DIR_JOB=${MYHOME}
cd $DIR_JOB'/python'

mpirun -np 1 python3 parallel_vec_mat_mult.py &> parallel_vec_mat_mult.out
