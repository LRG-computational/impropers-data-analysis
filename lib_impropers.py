import os
import re
import numpy as np
import pandas as pd

def find_missing_xyz_files(theta_range=None, phi_range=None):
    filename_pattern = re.compile(r"_Phi_(\d+)_Theta_(\d+)_")
    directories = ["XYZ_Files", "Hydrogenated_XYZ_Files", "Hydrogenated_Improper_XYZ_Files"]
    
    # If ranges are not provided, use a default range (0 to 360)
    if theta_range is None:
        theta_range = range(0, 360, 10)
    if phi_range is None:
        phi_range = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]

    missing_files = []

    for dir in directories:
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist.")
            continue

        # Create a set of all expected combinations
        expected_combinations = set(product(theta_range, phi_range))

        for file in os.listdir(dir):
            match = filename_pattern.search(file)
            if match:
                phi, theta = int(match.group(1)), int(match.group(2))
                if (theta, phi) in expected_combinations:
                    expected_combinations.remove((theta, phi))

        # Add remaining combinations as missing for this directory
        for theta, phi in expected_combinations:
            missing_files.append(f"{dir}: Missing Phi {phi}, Theta {theta}")

    print(f"Found {len(missing_files)} missing files.")
    return missing_files

# Define a function for parsing QChem outfiles with structured naming scheme
def extract_qchem_energies(path, filename_pattern, energy_pattern):
    """
    Extracts energy values from QChem output files located in a specified directory,
    matching a given filename pattern and energy line pattern.

    Parameters:
    - path (str): Path to the directory containing the output files.
    - filename_pattern (re.Pattern): Compiled regular expression pattern to match and extract
      phi and theta values from the filenames.
    - energy_pattern (re.Pattern): Compiled regular expression pattern to match and extract
      energy values from the file contents.

    Returns:
    - dict: A nested dictionary where the first level keys are phi values, the second level keys
      are theta values, and the values are the corresponding energy values extracted from the files.
    
    The function also prints a warning message for files where the energy value or phi and theta
    values cannot be extracted according to the given patterns.
    """
    # Initialize nested dictionary
    data = {}
    # Loop through each file in the directory
    for filename in os.listdir(path):
        if filename.endswith(".out"): # and filename.startswith("Me"):
            # Extract phi and theta from filename
            match = filename_pattern.search(filename)
            if match:
                phi = int(match.group(1))
                theta = int(match.group(2)) if match.group(2) else 0


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
    """
    Checks and prints the completeness of quantum mechanics (QM) calculations by analyzing
    the distribution of theta values for each phi value in the given energy dictionary.

    Parameters:
    - e_dict (dict): A nested dictionary where the first level keys are phi values, the second
      level keys are theta values, and the values are energy values.

    Returns:
    - None: This function only prints the sorted phi keys and the count of theta values for each
      phi, helping to visually inspect the completeness of QM calculations.
    """
    test = {}
    for phi in e_dict:
        test[phi]=[]
        for theta in e_dict[phi]:
            test[phi].append(theta)
    
    print(sorted(test.keys()))
    for i in sorted(test.keys()):
        print(i, len(test[i]))

# TODO: update this function to reflect the finalized calculation method
def calc_deloc_energies(total_dict, hyd_dict, meth_dict, conj_dict):
    """
    Calculates delocalization energies from the given energy dictionaries and creates a DataFrame
    with the results, including normalized methylene energies and new calculated values.

    Parameters:
    - total_dict (dict): Energy dictionary for total system.
    - hyd_dict (dict): Energy dictionary for the hydrogenated system.
    - meth_dict (dict): Energy dictionary for the methylated system.

    Returns:
    - pd.DataFrame: A DataFrame containing phi, theta, delocalization energy (E_deloc), nonbonded energy (E_nonbond),
      hydrogenated system energy (E_hyd), total system energy (E_tot), methylated system energy (E_meth),
      normalized methylated energy (norm_E_meth), new nonbonded energy (new_E_nb), and new delocalization energy (new_E_deloc)
      sorted by Theta and Phi.
    """
    data = []
    for phi in total_dict:

        # We want E_conjugation for each value of phi
        # E_conjugation is a scalar correction factor to the delocalization energy
        # for each value of phi
        # E_conjugation might be a separate dictionary, could just call the desired phi value when needed
        if phi <= 30:
            for theta in total_dict[phi]:
                E_nonbond = hyd_dict[phi][theta] - (meth_dict[phi][theta] - conj_dict[phi][0])
                E_deloc = (total_dict[phi][theta]) - E_nonbond 

                data.append([phi, theta, E_deloc, E_nonbond, hyd_dict[phi][theta], total_dict[phi][theta], meth_dict[phi][theta]])

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
    """
    Reads atomic coordinates from an XYZ file.

    Parameters:
    - filename (str): Path to the XYZ file.

    Returns:
    - list of tuples: A list where each tuple represents the x, y, and z coordinates of an atom.
    
    The function skips the first two lines of the XYZ file, which typically contain the number of atoms
    and a comment, respectively.
    """
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
    """
    Updates a LAMMPS data file with new atomic coordinates.

    Parameters:
    - lammps_file (str): Path to the original LAMMPS data file.
    - new_coords (list of tuples): New atomic coordinates to update in the data file, where each tuple contains the x, y, and z coordinates.
    - output_file (str): Path where the updated LAMMPS data file will be saved.

    Returns:
    - None: This function writes directly to a new file specified by `output_file`, replacing the atomic coordinates
      section with the new coordinates while preserving the rest of the data file structure.
    """
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