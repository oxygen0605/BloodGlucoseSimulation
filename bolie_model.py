# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:41:56 2021

@author: ozon0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp 

"""
一回のグルコース摂取量 g/l
 男性　一日平均接種量÷3  330g/3
 70 kgのとき 17.5 L
 血液に流れるグルコースの割合 50%
"""
q_eaten = 330/3/17.5*0.5




def f(t, x,a,b,c,d):
    i, g = x
    p = q = 0
    if (6 <= t < 7) or (12 <= t < 13) or (18 <= t < 19):
        q = q_eaten
        #print(t)
    dI_dt = p - a * i + b * g
    dG_dt = q - c * i - d * g
    return [dI_dt, dG_dt]


def drow_figs(t, g, i):
    
    g_label= r"G(t) blood glucose level [mmol/L]"
    i_label = r"I(t) blood insulin level [unit/L]"
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(t, g, 'C0', label = g_label) 
    
    ax2 = ax1.twinx()
    ln2 = plt.plot(t, i, 'C1',label = i_label)
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)
    
    ax1.set_xlabel('time [hours]')
    ax1.set_ylabel(g_label)
    ax1.set_xlim([0,24])
    #ax1.set_ylim([-10,60])
    ax1.grid(True)
    ax2.set_ylabel(i_label)
    ax2.set_xlim([0,24])
    #ax2.set_ylim([-0.01, 0.06])
    
    
    plt.show()

if __name__ == '__main__':
    dt = 0.01
    hr = 48
    #t = np.linspace(0, hr, int(hr/dt+1))
    t = np.arange(0, hr, dt)
    
    alpha = 0.916; beta = 0.198; gamma = 3.23; delta = 3.04
    d = (alpha+delta)**2 - 4*(alpha*delta + beta*gamma)
    print("D: {}".format(d))

    x0 = np.array([0.0, 0.0])
    solver = solve_ivp (f, [0,48], x0, t_eval=t, args=(alpha, beta, gamma, delta),dense_output=True, max_step=dt)
    
    g = solver.y[1,:]*100 # g/l -> mg/dL
    g /= 18 # mg/dL -> mmol/L
    i = solver.y[0, :]
    drow_figs(t, g, i)


"""
dt = 0.01
hr = 48
t = np.linspace(0, hr, int(hr/dt+1))
alpha = 0.916; beta = 0.198; gamma = 3.23; delta = 3.04

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
ln1 = ax1.plot(t, x[1, :], 'C0', label = "G(t) blood glucose level [mg/dL]") 

ax2 = ax1.twinx()
ln2 = plt.plot(t, x[0, :], 'C1',label = "I(t) blood insulin level [unit/L]")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)

ax1.set_xlabel('time [hours]')
ax1.set_ylabel(r'G(t) blood glucose level [mg/dL]')
ax1.set_xlim([0,24])
#ax1.set_ylim([-10,60])
ax1.grid(True)
ax2.set_ylabel(r'I(t) blood insulin level [unit/L]')
ax2.set_xlim([0,24])
#ax2.set_ylim([-0.01,0.06])

plt.show()

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ln1 = ax1.plot(x[0, :], x[1, :], 'C0', label = "blood glucose level [mg/dL]") 

ax1.set_xlabel(r'I(t) blood insulin level [unit/L]')
ax1.set_ylabel(r'G(t) blood glucose level [mg/dL]')
#ax1.set_xlim([-0.01,0.06])
#ax1.set_ylim([-10,60])
ax1.grid(True)
plt.show()
"""