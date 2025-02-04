import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
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

# Define a function for parsing QChem outfiles with structured naming scheme
def extract_qchem_energies(path, filename_pattern, energy_pattern):
    # Initialize nested dictionary
    if not isinstance(filename_pattern, re.Pattern) or not isinstance(energy_pattern, re.Pattern):
        raise ValueError("filename_pattern and energy_pattern must be compiled regular expressions")
    
    data = {}
    # Loop through each file in the directory
    for filename in os.listdir(path):
        if filename.endswith(".out"): # and filename.startswith("Me"):
            # Extract phi and theta from filename
            match = filename_pattern.search(filename)
            if match:
                phi = int(match.group(1))
                theta = int(match.group(2))
    
                # Read file and search for energy
                with open(os.path.join(path, filename), 'r') as f:
                    energy_found = False
                    for line in f:
                        energy_match = energy_pattern.search(line)
                        if energy_match:
                            energy = float(energy_match.group(1))
    
                            # Store in nested dictionary
                            if phi not in data:
                                data[phi] = {}
                            data[phi][theta] = energy
    
                            energy_found = True
                            break
    
                    # Throw a warning if energy is not found
                    if not energy_found:
                        print(f"Warning: Energy not found in file {filename}")
            else:
                print(f"Warning: Could not extract Phi and Theta values from filename {filename}")
    return data

def check_QM_completion(e_dict):
    test = {}
    for phi in e_dict:
        test[phi]=[]
        for theta in e_dict[phi]:
            test[phi].append(theta)
    
    print(sorted(test.keys()))
    for i in sorted(test.keys()):
        print(i, len(test[i]))
    return test

def calc_deloc_energies(total_dict, hyd_dict, meth_dict):
    """Calculates delocalization energies from total, hydrogenated, and methyl rotation energies.
    
    This function takes in dictionaries containing energies from QChem calculations 
    for the total system, the hydrogenated system, and the methylated system. It finds
    the minimum methyl rotation energy for each phi angle, subtracts that off to 
    normalize the methyl rotation energies for the given OOP angle, then calculates the nonbonded and 
    delocalization energies. The results are returned as a Pandas DataFrame.
    """
    data = []
    for phi in total_dict:
        E_meth_min = np.min([meth_dict[phi][t] for t in meth_dict[phi]])
        for theta in total_dict[phi]:
            # if theta >= 30:
                # print(phi,theta)
            E_nonbond = hyd_dict[phi][theta] - (meth_dict[phi][theta] - E_meth_min)
            E_deloc = total_dict[phi][theta] - E_nonbond

            data.append([phi, theta, E_deloc, E_nonbond, hyd_dict[phi][theta], total_dict[phi][theta], (meth_dict[phi][theta] - E_meth_min)])

    df = pd.DataFrame(data, columns=['Phi', 'Theta', 'E_deloc', 'E_nonbond', 'E_hyd', 'E_tot', 'E_meth'])
    sorted_df = df.sort_values(['Theta', 'Phi'])
    
    new_e_meth = []
    for phi in sorted_df['Phi'].unique():
        subset = sorted_df[sorted_df['Phi'] == phi]
        min = np.min(subset['E_meth'])
        norm = np.array(subset['E_meth'] - min)
        new_e_meth.append(norm)
    
    sorted_df['norm_E_meth'] = [i for sub in new_e_meth for i in sub]
    sorted_df['new_E_nb'] = sorted_df['E_hyd'] - sorted_df['norm_E_meth']
    sorted_df['new_E_deloc'] = sorted_df['E_tot'] - sorted_df['new_E_nb']
    
    return sorted_df

# Function to read XYZ coordinates
def read_xyz(filename):
    coords = []
    with open(filename, 'r') as f:
        for _ in range(2):  # Skip the first two lines
            next(f)
        for line in f:
            _, x, y, z = line.split()
            coords.append((float(x), float(y), float(z)))
    return coords


# Function to update LAMMPS data file with new coordinates
def update_lammps_data(lammps_file, new_coords, output_file):
    with open(lammps_file, 'r') as f_in, open(output_file, 'w') as f_out:
        atom_section = False
        i = 0  # Index for new_coords
        for line in f_in:
            if "Atoms" in line:                          
                atom_section = True
                f_out.write(line)
                next(f_in)  # Skip the empty line after "Atoms"
                f_out.write("\n")
                continue
            elif "Bonds" in line or "Velocities" in line:
                atom_section = False

            if atom_section:
                parts = line.split()
                if len(parts) == 7:  # id molecule_type atom_type charge x y z
                    new_line = f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\t{new_coords[i][0]:.6f}\t{new_coords[i][1]:.6f}\t{new_coords[i][2]:.6f}\n"
                    f_out.write(new_line)
                    i += 1
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

def show_energy_barriers(molecules_data):
    colors = ['orange', 'green', 'red', 'blue']
    plt.figure(figsize=(8, 6))

    for (sorted_deloc, phis, name), color in zip(molecules_data, colors):
        barriers = []
        phis_used = []
        for phi in phis:
            if phi <= 30:
                subset = sorted_deloc[sorted_deloc['Phi'] == phi]
                e_conjugation = subset[subset['Theta'] == 0]['E_meth'].iloc[0]
                
                max1 = subset['E_deloc'][subset['Theta'] == 90].values[0]
                max2 = subset['E_deloc'][subset['Theta'] == 270].values[0]
                min = subset['E_deloc'][subset['Theta'] == 0].values[0]
                barrier = ((max1 - min) + (max2 - min)) / 2

                phis_used.append(phi)
                barriers.append(barrier * 627.509)  # Normalization factor for energy

        plt.plot(phis_used, barriers, color=color, label=name, linewidth=4) 
        plt.scatter(phis_used, barriers, color=color, edgecolor='black', facecolor=color, marker='s',s=100, linewidth=1.5)

    plt.xlabel('Improper Angle (degrees)')
    plt.ylabel('Torsional Barrier (kcal/mol)')
    plt.title('Torsional Barriers for Multiple Molecules (Phi <= 30)')
    plt.legend()
    plt.show()

def show_energy_barriers_kj_mol(molecules_data):
    colors = ['orange', 'green', 'red', 'blue']
    plt.figure(figsize=(8, 6))

    conversion_factor = 2625.5

    for (sorted_deloc, phis, name), color in zip(molecules_data, colors):
        barriers = []
        phis_used = []
        for phi in phis:
            if phi <= 30:
                subset = sorted_deloc[sorted_deloc['Phi'] == phi]
                e_conjugation = subset[subset['Theta'] == 0]['E_meth'].iloc[0]
                
                max1 = subset['E_deloc'][subset['Theta'] == 90].values[0]
                max2 = subset['E_deloc'][subset['Theta'] == 270].values[0]
                min_energy = subset['E_deloc'][subset['Theta'] == 0].values[0]
                barrier = ((max1 - min_energy) + (max2 - min_energy)) / 2

                phis_used.append(phi)
                barriers.append(barrier * conversion_factor)  

        plt.plot(phis_used, barriers, color=color, label=name, linewidth=4)
        plt.scatter(phis_used, barriers, color=color, edgecolor='black', facecolor=color, marker='s', s=100, linewidth=1.5)

    plt.xlabel('Improper Angle (degrees)')
    plt.ylabel('Torsional Barrier (kJ/mol)')
    plt.title('Torsional Barriers for Multiple Molecules (Phi <= 30)')
    plt.legend()
    plt.show()

import pandas as pd
import numpy as np

# Constants
conversion_factor = 627.509
reference_energy = -629.5025  # Energy in Hartree, adjust as necessary
print ((molecules_data[0][0]['E_tot'] - (molecules_data[0][0]['E_hyd'] - molecules_data[0][0]['E_meth'])))
# Assuming molecules_data is populated as shown in your earlier message
for sorted_deloc, phis, name in molecules_data:
    rows = []
    for phi in phis:
        subset = sorted_deloc[sorted_deloc['Phi'] == phi]
        min_e = np.min(subset['E_deloc'])  # Assuming minimum energy calculation is per Phi subset
        for theta in subset['Theta'].unique():
            subset_theta = subset[subset['Theta'] == theta]
            if not subset_theta.empty:
                e_deloc = subset_theta['E_deloc'].iloc[0]
                # Assuming 'E_tot', 'E_hyd', 'E_meth' are columns in your data
                energy = subset_theta['E_tot'].iloc[0] - (subset_theta['E_hyd'].iloc[0] - subset_theta['E_meth'].iloc[0])
                adjusted_energy = conversion_factor * (energy - reference_energy)
                rows.append({'Phi': phi, 'Theta': theta, 'E_deloc': adjusted_energy})
    
    # Create DataFrame from rows
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = f'{name}.csv'
    df.to_csv(csv_path, index=False)
    print(f'CSV file saved: {csv_path}')
