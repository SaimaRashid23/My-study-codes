# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:33:11 2025

@author: saima
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages

# Set fractional order
alpha = 0.95

# Define compartments with fractional scaling (Replace with actual fractional model if available)
np.random.seed(42)
S = np.random.gamma(5 * alpha, 50000 * alpha, 1000)  # Susceptible
V = np.random.gamma(4 * alpha, 30000 * alpha, 1000)  # Vaccinated
A = np.random.gamma(3 * alpha, 20000 * alpha, 1000)  # Asymptomatic
I = np.random.gamma(2 * alpha, 15000 * alpha, 1000)  # Infected
M = np.random.gamma(6 * alpha, 40000 * alpha, 1000)  # Recovered

data = {'S': S, 'V': V, 'A': A, 'I': I, 'M': M}
labels = ['Susceptible (S)', 'Vaccinated (V)', 'Asymptomatic (A)', 'Infected (I)', 'Recovered (M)']
colors = ['b', 'g', 'r', 'm', 'c']

# Create a PDF file for saving the plots
with PdfPages("fractional_relative_freq_pdf_plots_alpha_095.pdf") as pdf:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
    axes = axes.ravel()  # Flatten the 2D array of axes into a 1D array
    
    for i, (key, label, color) in enumerate(zip(data.keys(), labels, colors)):
        ax = axes[i]  # Assign current subplot
        
        # Compute relative frequency histogram
        sns.histplot(data[key], bins=30, stat='density', kde=False, color=color, edgecolor='black', alpha=0.6, ax=ax)
        
        # Compute and plot Probability Density Function (PDF)
        kde = gaussian_kde(data[key])
        x_vals = np.linspace(min(data[key]), max(data[key]), 1000)
        ax.plot(x_vals, kde(x_vals), color='k', linestyle='dashed', linewidth=3, label=f'PDF (α={alpha})')
        
        # Formatting
        ax.set_title(f"{label} (α={alpha})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Value", fontsize=12, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

    # Remove the last (empty) subplot since we have only 5 compartments
    fig.delaxes(axes[-1])

    plt.tight_layout()
    pdf.savefig()
    plt.show()
