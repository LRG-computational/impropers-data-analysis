
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import pandas as pd
from scipy.optimize import curve_fit


"""This is the script used to generate Figures 7 and 8 (absolute improper torsional barrier and improper/dihedral comparison, respectively)"""

def OPLS(x,a,b,c,d,e):
    #Returns an OPLS measurement or fitting, potentially offset in y-direction, of how the rings move
    return .5*a*(1+np.cos(np.radians(x)))+.5*b*(1-np.cos(np.radians(2*x)))+.5*c*(1+np.cos(np.radians(3*x)))+.5*d*(1-np.cos(np.radians(4*x)))+e

def OPLS_Derivative(x,a,b,c,d):
    #Returns the first derivative of the OPLS function
    return .5*a*(-np.sin(np.radians(x))) + b*(np.sin(np.radians(2*x))) + 1.5*c*(-np.sin(np.radians(3*x))) + 2*d*(np.sin(np.radians(4*x)))

def fourier_nonfit(x, a_params, b_params, a0, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    series = a0
    for a,b,i in zip(a_params,b_params,range(1,n+1)):
        series += a * cos(i * x) + b * sin(i *  x)
    return series

def fourier_series_derivative(x, a_params, b_params, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms

    # Construct the series
    series = 0
    for a,b,i in zip(a_params,b_params,range(1,n+1)):
        series += -1 * a * i * sin(i * x) + b * i * cos(i * x)
    return series

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))[0],parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))[1:]
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


import pickle

molecules_data = []
with open('./ptb7out/data.pkl', 'rb') as file:
    data = pickle.load(file)
    sorted_deloc = data['sorted_deloc']
    phis = data['phis']
    molecules_data.append((sorted_deloc, phis, "PTB7OUT"))

with open('./ptb7in/data.pkl', 'rb') as file:
    data = pickle.load(file)
    sorted_deloc = data['sorted_deloc']
    phis = data['phis']
    molecules_data.append((sorted_deloc, phis, "PTB7IN"))

with open('./pndit/data.pkl', 'rb') as file:
    data = pickle.load(file)
    sorted_deloc = data['sorted_deloc']
    phis = data['phis']
    molecules_data.append((sorted_deloc, phis, "PNDIT"))

with open('./p3ht/data.pkl', 'rb') as file:
    data = pickle.load(file)
    sorted_deloc = data['sorted_deloc']
    phis = data['phis']
    molecules_data.append((sorted_deloc, phis, "P3HT"))

# Will pass in Molecules data for Dimer_Delocalization Energies
# Rotated_Shape??


# Function with Shruti's commentary and modifications
def Fit_Improper_Energies(Dimer_Delocalization_Energies,Rotated_Shape):
    """
    Fits the energies to a 6th order fourier series and graphs them
    : param Dimer_Delocalization Energies: energies of all the dimers
    : param Rotated_Shape: 
    """
    
    color = np.linspace(0.0,1.0,Rotated_Shape[1])
    cmap = mpl.cm.get_cmap("coolwarm")

    Barriers = []

    for deloc_matrix,_,_,_,_ in zip(Dimer_Delocalization_Energies):
        deloc_matrix = deloc_matrix - np.amin(deloc_matrix)
        Parameter_Lists = []
        fig,ax = plt.subplots(1,1)
        for i,energy_list in enumerate(deloc_matrix):
            if np.mod(i,7) == 0:
                alpha = 1.0
            else:
                alpha = 0.2
            ax.scatter(np.linspace(0,350,Rotated_Shape[1]),energy_list,alpha=alpha,marker = 's',c=cmap(color[i]))
            
            # opt is the optimal values of the parameters so that the sum of the squared residuals of OPLS(np.linspace(0,350,Rotated_Shape[1]), *opt) - energy_list is minimized.
            opt,_ = scipy.optimize.curve_fit(OPLS,np.linspace(0,350,Rotated_Shape[1]),energy_list)
            
            Parameter_Lists.append(opt)
            x = np.linspace(0,360,10000)
            fit = []
            for j in x:
                fit.append(OPLS(j,opt[0],opt[1],opt[2],opt[3],opt[4]))
            Barriers.append(fit[2500] - fit[0])
            plt.plot(x,fit,color=cmap(color[i]),alpha=alpha)

        plt.xlabel('Dihedral Angle ($^\circ$)',size = 24)
        plt.ylabel('Energy (kcal/mol)',size = 24)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(length=4,width=4)

        plt.tight_layout()
        fig.savefig('figure')

def fourier_series1(x, *coefficients):
    n = len(coefficients) // 2
    result = coefficients[0]  
    for i in range(1, n):
        a, b = coefficients[2*i-1], coefficients[2*i]
        result += a * np.cos(i * np.pi * x / 180) + b * np.sin(i * np.pi * x / 180)
    return result

def graph_molecule_data2(molecules_data, rotated_shape):
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = mpl.colormaps["coolwarm"]
    color = np.linspace(0.0, 1.0, rotated_shape[1])


    for index, data in enumerate(molecules_data):
        if index > 3:
            break

        if data:
            try:
                data = data[0]['E_deloc']
                x = np.linspace(0, 350, len(data))  
                y = np.array(data) 
                min_value = np.min(y) 
                y = y - min_value 

                ax.scatter(x, y, color=cmap(color[index % len(color)]), label=f"Molecule {index + 1}")
            except Exception as e:
                print(f"Skipping dataset at index {index} due to error: {e}")
                continue

    ax.set_xlabel('Dihedral Angle ($^\circ$)', size=24)
    ax.set_ylabel('Energy (kcal/mol)', size=24)
    if index > -1:  
        ax.legend()
    else:
        print("No data was suitable for plotting.")

    plt.tight_layout()
    plt.show()

def graph_molecule_data3(molecules_data, rotated_shape, index):
    fig, ax = plt.subplots(figsize=(7, 8))
    cmap = mpl.colormaps["coolwarm"]
    color = np.linspace(0.0, 1.0, rotated_shape[1])


    data = molecules_data[index]

    if data:
        try:
            data = data[index]['E_deloc']
            x = np.linspace(0, 350, len(data))
            y = np.array(data)
            min_value = np.min(y)
            y = y - min_value
            
            labels = ['ptb7out', 'ptb7in', 'pndit', 'p3ht']
            label = labels[index] if index < len(labels) else f"Molecule {index + 1}"
            ax.scatter(x, y, color=cmap(color[0 % len(color)]), label=label)

            # 6 for 6th order
            guess = [0] * (2 * 6 + 1)  
            params, _= curve_fit(lambda x, *params: fourier_series1(x, *params),x, y, p0=guess)

            x_fit = np.linspace(0, 350, 400)
            y_fit = fourier_series1(x_fit, *params)
            ax.plot(x_fit, y_fit, color='black', alpha=0.7, linewidth=2)
            
        except Exception as e:
            print(f"Skipping dataset at index {index} due to error: {e}")

    ax.set_xlabel('Dihedral Angle ($^\circ$)', size=24)
    ax.set_ylabel('Energy (kcal/mol)', size=24)
    
    plt.tight_layout()
    plt.show()

graph_molecule_data3(molecules_data, (0, 10),0)