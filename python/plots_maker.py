#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Get the filename from command line argument without extension
filename = Path(sys.argv[1]) 

infos = filename.stem.split('_')

n = infos[1]  # Assuming the second part is the size of the matrix
name = '_'.join(infos[2:-1])  # Assuming the central part is the program name
language = infos[-1]  # Assuming the last part is the prog language 

prog_name = '.'.join((name, language))  # Combine name and language for the program name

#extension =  '.' + language

ntasks = []
times = []
speedup = []
with open(filename, 'r') as f:
    content = f.readlines()

for line in content:
    string = line.strip().split()
    n, time = int(string[0]), float(string[1])
    ntasks.append(int(n))
    times.append(float(time))

# sorting by n of tasks
times = [t for _, t in sorted(zip(ntasks, times))]
speedup = [times[0] / t for t in times]  # Calculate speedup based on the first time
ntasks.sort()

fig_scal, ax_scal = plt.subplots(figsize=(8, 6))
ax_scal.plot(ntasks, times, marker='o')
ax_scal.set_xlabel('Number of tasks')
ax_scal.set_ylabel('Computing Time (s)')
ax_scal.set_title(f'Scaling of \'{prog_name}\' - N={n}')
ax_scal.grid(True)
fig_scal.savefig(f'scaling_plot_{name}_{language}_{n}.png')

fig_sp, ax_sp = plt.subplots(figsize=(8, 6))
ax_sp.plot(ntasks, speedup, marker='o', label='Speedup')
ax_sp.plot(ntasks, ntasks, marker='s', label='Ideal Speedup', linestyle='--')
ax_sp.set_xlabel('Number of tasks')
ax_sp.set_ylabel('Speedup')
ax_sp.set_title(f'Speedup of \'{prog_name}\' - N={n}')
ax_sp.grid(True)
#ax_sp.sp
fig_scal.savefig(f'speedup_plot_{name}_{language}_{n}.png')

plt.show()

for n,t in zip(ntasks, times):
    print(n,t)
