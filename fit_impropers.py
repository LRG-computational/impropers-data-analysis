
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.colors import Normalize



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

#graph_molecule_data3(molecules_data, (0, 10),0)

# Define the Fourier series function
def fourier_series(theta, *coefficients):
    a0 = coefficients[0]
    result = a0 / 2  # a0/2 represents the constant offset
    n_harmonics = len(coefficients[1:]) // 2
    for n in range(n_harmonics):
        a_n = coefficients[2*n + 1]
        b_n = coefficients[2*n + 2]
        result += a_n * np.cos((n + 1) * theta) + b_n * np.sin((n + 1) * theta)
    return result


# Your existing plotting function modified to include Fourier series fitting
def plot_with_fourier_fits(rimp2_df, phis, cmap, norm, axes, title):
    min_e = np.min(rimp2_df[0]['E_deloc'])
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, ax in enumerate(axs.flatten()):
        for phi in phis:
            if phi <= 30:
                subset = rimp2_df[0][rimp2_df[0]['Phi'] == phi]
                theta = np.radians(subset['Theta'])  # Convert to radians if theta is in degrees
                if i == 0:
                    energy = 627.509 * (subset['E_deloc'] - min_e)
                else:
                    energy = subset['E_tot'] - (subset['E_hyd'] - subset['E_meth'])
                    energy = 627.509 * (energy - -629.5025)
                
                color = cmap(norm(phi))
                ax.scatter(subset['Theta'], energy, color=color, label=f"Phi={phi}")

                # Fit the Fourier series
                n_harmonics = 6  # Define the number of harmonics
                coeffs_guess = [0] * (1 + 2 * n_harmonics)  # Initial guesses (a0, a1, b1, ..., an, bn)
                fitted_params, _ = curve_fit(fourier_series, theta, energy, p0=coeffs_guess)

                # Generate the fitted curve
                theta_fitted = np.linspace(min(theta), max(theta), 200)
                fitted_energy = fourier_series(theta_fitted, *fitted_params)

                # Convert radians back to degrees if needed
                theta_fitted_degrees = np.degrees(theta_fitted)
                ax.plot(theta_fitted_degrees, fitted_energy, color='black', linewidth=2, linestyle='--', label='Fitted Curve')

        ax.set_xlabel('Theta (degrees)', fontdict=axes)
        ax.set_ylabel('Energy (arbitrary units)', fontdict=axes)
        ax.set_title(f'Plot {i+1}: Energy vs Theta for Phi <= 30', fontdict=title)
        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

# Assuming you have the required data and variables set up, call the function

phis = [0,5,10,15,20,25,30]

axes = {
    'fontname': 'Arial',  
    'size': 12,           
    'weight': 'bold'      
}

title = {
    'fontname': 'Arial',
    'size': 14,
    'weight': 'bold'
}
cmap = plt.cm.viridis

min_phi = min(phis) 
max_phi = max(phis)
norm = Normalize(vmin=min_phi, vmax=max_phi)


def opls_dihedral(theta, V1, V2, V3):
    """ OPLS dihedral potential function """
    return (V1 / 2) * (1 + np.cos(theta)) + \
           (V2 / 2) * (1 - np.cos(2 * theta)) + \
           (V3 / 2) * (1 + np.cos(3 * theta))

def plot_with_opls_fits(rimp2_df, phis, cmap, norm, axes, title):
    min_e = np.min(rimp2_df[0]['E_deloc'])
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, ax in enumerate(axs.flatten()):
        for phi in phis:
            if phi <= 30:
                subset = rimp2_df[0][rimp2_df[0]['Phi'] == phi]
                theta = np.radians(subset['Theta'])  # Convert to radians if theta is in degrees
                if i == 0:
                    energy = 627.509 * (subset['E_deloc'] - min_e)
                else:
                    energy = subset['E_tot'] - (subset['E_hyd'] - subset['E_meth'])
                    energy = 627.509 * (energy - -629.5025)
                
                color = cmap(norm(phi))
                ax.scatter(subset['Theta'], energy, color=color, label=f"Phi={phi}")

                # Fit the OPLS dihedral function
                coeffs_guess = [0, 0, 0]  # Initial guesses (V1, V2, V3)
                fitted_params, _ = curve_fit(opls_dihedral, theta, energy, p0=coeffs_guess)

                # Generate the fitted curve
                theta_fitted = np.linspace(min(theta), max(theta), 200)
                fitted_energy = opls_dihedral(theta_fitted, *fitted_params)

                # Convert radians back to degrees if needed
                theta_fitted_degrees = np.degrees(theta_fitted)
                ax.plot(theta_fitted_degrees, fitted_energy, color='black', linewidth=2, linestyle='--', label='Fitted Curve')

        ax.set_xlabel('Theta (degrees)', fontdict=axes)
        ax.set_ylabel('Energy (arbitrary units)', fontdict=axes)
        ax.set_title(f'Plot {i+1}: OPLS Energy vs Theta for Phi <= 30', fontdict=title)
        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()


# ptb7out
# plot_with_fourier_fits(molecules_data[0], phis, cmap, norm, axes, title)
# ptb7in
# plot_with_fourier_fits(molecules_data[1], phis, cmap, norm, axes, title)
# pndit
# plot_with_fourier_fits(molecules_data[2], phis, cmap, norm, axes, title)
# p3ht
# plot_with_fourier_fits(molecules_data[3], phis, cmap, norm, axes, title)

plot_with_opls_fits(molecules_data[0], phis, cmap, norm, axes, title)