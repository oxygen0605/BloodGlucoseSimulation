# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:41:56 2021

@author: ozon0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp 
from math import sqrt, pi, exp
"""
一回のグルコース摂取量 g/l
 男性　一日平均接種量÷3  330g/3
 70 kgのとき 17.5 L
 血液に流れるグルコースの割合 50%
"""

q_eaten = 330.0/3.0/17.5*0.5# 一回の食事あたりにされるグルコース量　[g/l]

p_eq = 0.006 # insulinの平衡点
q_eq = 0.9   # グルコースの平衡点

def f(t, x,a,b,c,d):
    i, g = x
    q = c*p_eq + d*q_eq  # gamma*0.006 + delta*0.9
    p = a*p_eq - b*q_eq  # alpha*0.006 - beta *0.9

    #if (6 <= t < 8) or (12 <= t < 14) or (18 <= t < 20):
    #    q += 0.5*q_eaten
    sig = 0.67
    if (6 <= t < 24):
        q += q_eaten*(exp(-(t-6.5)/(2*sig**2))/(sqrt(2*pi*sig**2)))
    if (12 <= t < 24):
        q += q_eaten*(exp(-(t-12.5)/(2*sig**2))/(sqrt(2*pi*sig**2)))
    if (18 <= t < 24):
        q += q_eaten*(exp(-(t-18.5)/(2*sig**2))/(sqrt(2*pi*sig**2)))
        
    dI_dt = p - a * i + b * g
    dG_dt = q - c * i - d * g
    return [dI_dt, dG_dt]


def plot_glu_and_ins(t, g, i, unit="mg/dl"):
    
    g_label= r"G(t) blood glucose level [{}]".format(unit)
    i_label = r"I(t) blood insulin level [unit/L]"
    
    # new figure
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # x軸の設定
    ax1.set_xlabel('time [hours]')
    ax1.set_xlim([0,24])
    
    # 一つ目y軸 (glucose)の設定
    ln1 = ax1.plot(t, g, 'C0', label = g_label)
    ax1.set_ylabel(g_label)
    #ax1.set_ylim([-10,200]) if unit == "mg/dl" else ax1.set_ylim([-10,20])
    ax1.grid(True)
    
    # 二つ目y軸 (insulin)の設定
    ax2 = ax1.twinx()
    ln2 = plt.plot(t, i, 'C1',label = i_label)
    ax2.set_ylabel(i_label)
    ax2.set_xlim([0,24])
    #ax2.set_ylim([-0.01, 1.0])
    
    
    # legendの設定
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)
    
    plt.show()
    


if __name__ == '__main__':

    
    # 初期値、パラメータ設定
    dt = 0.01; hr = 48
    t = np.arange(0, hr, dt)
    alpha = 0.916; beta = 0.198; gamma = 3.23; delta = 3.04
    d = (alpha+delta)**2 - 4*(alpha*delta + beta*gamma)
    print("固有値の判別式 D: {}".format(d))
    x0 = np.array([0.006, 0.9])
    
    # 数値計算
    solver = solve_ivp (f, [0,48], x0, t_eval=t, args=(alpha, beta, gamma, delta),dense_output=True, max_step=dt)
    
    # 単位変換＆グラフプロット
    g = solver.y[1,:]*100 # g/l -> mg/dL
    plot_glu_and_ins(t, g, i)
    
    # 単位変換＆グラフプロット
    g /= 18 # mg/dL -> mmol/L
    i = solver.y[0, :]
    plot_glu_and_ins(t, g, i, "mmol/L")


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