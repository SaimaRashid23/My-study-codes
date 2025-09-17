# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:16:27 2025

@author: saima
"""
import numpy as n
import matplotlib.pyplot as plt
def euler_maruyama(dt, T, params, control=False):
    N = int(T/dt)  # Number of time steps
    t = np.linspace(0, T, N)
    
    # Initialize compartments
    U, V, W, X, Y = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    
    # Initial conditions
    U[0], V[0], W[0], X[0], Y[0] = params['U_0'], params['V_0'], params['W_0'], params['X_0'], params['Y_0']
    
    # Stochastic noise coefficients
    wp1, wp2, wp3, wp4, wp5 = params['wp1'], params['wp2'], params['wp3'], params['wp4'], params['wp5']
    
    for k in range(1, N):
        dW = np.sqrt(dt) * np.random.randn()  # Wiener process increment
        
        # Stochastic differential equations
        U[k] = U[k-1] + dt * (params['B1'] - params['Gamma1'] * U[k-1] - params['phi1'] * U[k-1] \
                + params['theta1'] * V[k-1] - params['d1'] * U[k-1]) + wp1 * U[k-1] * dW
        
        V[k] = V[k-1] + dt * (params['phi1'] * U[k-1] - params['theta1'] * V[k-1] - params['sigma1'] * params['Gamma1'] * V[k-1] \
                - params['d1'] * V[k-1]) + wp2 * V[k-1] * dW
        
        W[k] = W[k-1] + dt * ((1 - params['p1']) * params['Gamma1'] * U[k-1] - params['delta1'] * W[k-1] \
                - params['d1'] * W[k-1]) + wp3 * W[k-1] * dW
        
        X[k] = X[k-1] + dt * (params['p1'] * params['Gamma1'] * U[k-1] + params['sigma1'] * params['Gamma1'] * V[k-1] \
                - (params['eta1'] + params['mu1']) * X[k-1] - params['d1'] * X[k-1]) + wp4 * X[k-1] * dW
        
        Y[k] = Y[k-1] + dt * (params['eta1'] * X[k-1] - params['omega1'] * Y[k-1] - params['d1'] * Y[k-1]) \
                + wp5 * Y[k-1] * dW
    
    return t, U, V, W, X, Y

# Parameters
params = {
    'B1': 500, 'Gamma1': 0.02, 'phi1': 0.01, 'theta1': 0.03, 'sigma1': 0.02,
    'p1': 0.1, 'delta1': 0.01, 'eta1': 0.02, 'mu1': 0.01, 'omega1': 0.01,
    'd1': 0.005, 'wp1': 0.1, 'wp2': 0.2, 'wp3': 0.3, 'wp4': 0.4, 'wp5': 0.5,
    'U_0': 500000, 'V_0': 200000, 'W_0': 10000, 'X_0': 5000, 'Y_0': 2000
}

T, dt = 30, 0.1

# Solve for both cases
t, U_no_control, V_no_control, W_no_control, X_no_control, Y_no_control = euler_maruyama(dt, T, params, control=False)
t, U_control, V_control, W_control, X_control, Y_control = euler_maruyama(dt, T, params, control=True)

# Plot results
fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

axs[0, 0].plot(t, U_no_control, label='U (No Control)', color='blue', linewidth=3)
axs[0, 0].plot(t, U_control, '--', label='U (Control)', color='red', linewidth=3)
axs[0, 0].set_ylabel('U', fontsize=14)
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(t, V_no_control, label='V (No Control)', color='blue', linewidth=3)
axs[0, 1].plot(t, V_control, '--', label='V (Control)', color='red', linewidth=3)
axs[0, 1].set_ylabel('V', fontsize=14)
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[0, 2].plot(t, W_no_control, label='W (No Control)', color='blue', linewidth=3)
axs[0, 2].plot(t, W_control, '--', label='W (Control)', color='red', linewidth=3)
axs[0, 2].set_ylabel('W', fontsize=14)
axs[0, 2].legend()
axs[0, 2].grid(True)

axs[1, 0].plot(t, X_no_control, label='X (No Control)', color='blue', linewidth=3)
axs[1, 0].plot(t, X_control, '--', label='X (Control)', color='red', linewidth=3)
axs[1, 0].set_ylabel('X', fontsize=14)
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(t, Y_no_control, label='Y (No Control)', color='blue', linewidth=3)
axs[1, 1].plot(t, Y_control, '--', label='Y (Control)', color='red', linewidth=3)
axs[1, 1].set_ylabel('Y', fontsize=14)
axs[1, 1].set_xlabel('Time', fontsize=14)
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.delaxes(axs[1, 2])

plt.tight_layout()
plt.savefig("output.pdf", format="pdf", dpi=300)
plt.show()

