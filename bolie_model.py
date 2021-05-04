# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:41:56 2021

@author: ozon0
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
hr = 48
t = np.linspace(0, hr, int(hr/dt+1))

alpha = 0.916
beta =  0.198
gamma = 3.23
delta = 3.04

p = np.zeros((1, int(hr/dt+1)))
q = np.zeros((1, int(hr/dt+1)))

#q[0, 0:1] = 20
q[0, 601:700] = 100/60
#q[0, 1201:1300] = 100/60
#q[0, 1801:1900] = 100/60


A = np.array([[1-alpha*dt, beta*dt], [-gamma*dt, 1-delta*dt]])
P = np.concatenate((p, q), axis = 0) * dt

x = np.empty((2, int(hr/dt+1)))
for i in range(int(hr/dt)):
    x[:, (i+1)] = np.dot(A, x[:, i]) + P[:, i]

x[1, :] = x[1, :] * 100

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(t, x[1, :], 'C0', label = "blood glucose level [mg/dL]") 

ax2 = ax1.twinx()
ln2 = plt.plot(t, x[0, :], 'C1',label = "blood insulin level [unit/L]")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)

ax1.set_xlabel('time [hours]')
ax1.set_ylabel(r'blood glucose level [mg/dL]')
ax1.set_xlim([0,24])
ax1.set_ylim([-10,60])
ax1.grid(True)
ax2.set_ylabel(r'blood insulin level [unit/L]')
ax2.set_xlim([0,24])
ax2.set_ylim([-0.01,0.06])

plt.show()

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ln1 = ax1.plot(x[0, :], x[1, :], 'C0', label = "blood glucose level [mg/dL]") 

ax1.set_xlabel(r'blood insulin level [unit/L]')
ax1.set_ylabel(r'blood glucose level [mg/dL]')
#ax1.set_xlim([-0.01,0.06])
#ax1.set_ylim([-10,60])
ax1.grid(True)
plt.show()

"""
x = np.arange(-5, 50)
y = np.arange(0.0, 0.06, 0.001)
P = np.concatenate((x, y), axis = 0) * dt
u = [-gamma*dt, 1-delta*dt]*[y,x]
v = [1-alpha*dt, beta*dt]

x, y = np.meshgrid(x, y)
u, v = np.meshgrid(u, v)
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
 
lim = 8
for ax in axes:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
 
    ax.set_xticks(np.arange(-lim, lim, 1))
    ax.set_yticks(np.arange(-lim, lim, 1))
 
    ax.grid()
    ax.set_aspect('equal')
 
C = np.sqrt(u * u + v * v)
axes[0].quiver(x, y, u, v)
axes[1].quiver(x, y, u, v, C, scale=100, cmap='Blues')
 
plt.show()
"""