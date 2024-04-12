#! /usr/local/opt/python@2/bin/python2.7

import numpy as np
import Atom
import Molecule
import Ring
import OPLS as op
import System
import Conjugated_Polymer
import Cluster_IO
import Write_Inputs
import Write_Submit_Script
import math
import copy
import scipy
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib as mpl
import re
import scipy
from symfit import parameters, variables, sin, cos, Fit

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

def Fit_Improper_Energies(Dimer_Delocalization_Energies,Ring_List,Rotated_Shape,Merged_OOP_Rotations,Dimer_Weights,Coarse_Grid_Size=(13,72),Nonbonded=False):
    color = np.linspace(0.0,1.0,Rotated_Shape[1])
    cmap = mpl.cm.get_cmap("coolwarm")
    Offset_Ring_List = []

    Merged_OOP_Rotations_Radians = np.array([np.radians(x) for x in Merged_OOP_Rotations])

    a_params_list = []
    b_params_list = []
    a0_params_list = []

    for ring in Ring_List[1:]:
        Offset_Ring_List.append(ring)

    Offset_Ring_List.append(Ring_List[0])

    Fit_Energies_list = []
    Fit_Coarse_Energies_list = []
    Force_x_list = []
    Force_y_list = []

    Ring1_List = []
    Ring2_List = []

    for ring1,ring2 in zip(Ring_List,Offset_Ring_List):
        Ring1_List.append(ring1)
        Ring1_List.append(ring2)
        Ring2_List.append(ring2)
        Ring2_List.append(ring1)

    Barriers = []

    for deloc_matrix,ring1,ring2,weights,z in zip(Dimer_Delocalization_Energies,Ring1_List,Ring2_List,Dimer_Weights,range(len(Ring1_List))):
        deloc_matrix = deloc_matrix - np.amin(deloc_matrix)
        Parameter_Lists = []
        fig,ax = plt.subplots(1,1)
        for i,energy_list in enumerate(deloc_matrix):
            if np.mod(i,7) == 0:
                alpha = 1.0
            else:
                alpha = 0.2
            ax.scatter(np.linspace(0,350,Rotated_Shape[1]),energy_list,alpha=alpha,marker = 's',c=cmap(color[i]))
            opt,cov = scipy.optimize.curve_fit(OPLS,np.linspace(0,350,Rotated_Shape[1]),energy_list)
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
        #plt.ylim([-1,15])
        plt.tight_layout()
        if Nonbonded:
            fig.savefig('%s_%s_Conjugated_Energies_Nonbonded' % (ring1.Name,ring2.Name))
            plt.close(fig)
            os.system("scp %s_%s_Conjugated_Energies_Nonbonded.png ./Figures" % (ring1.Name,ring2.Name))
            os.system("rm -f %s_%s_Conjugated_Energies_Nonbonded.png" % (ring1.Name,ring2.Name))
        else:
            fig.savefig('%s_%s_Conjugated_Energies' % (ring1.Name,ring2.Name))
            plt.close(fig)
            os.system("scp %s_%s_Conjugated_Energies.png ./Figures" % (ring1.Name,ring2.Name))
            os.system("rm -f %s_%s_Conjugated_Energies.png" % (ring1.Name,ring2.Name))

        opls_params = []
        a_params = []
        b_params = []
        a0_params = []
        fourier_order = 6
        for j in range(len(opt)):
            x, y = variables('x, y')
            w, = parameters('w')
            #model_dict = {y: fourier_series(x, f=w, n=6)}
            model_dict = {y: fourier_series(x, f=1, n=fourier_order)}
            fig,ax = plt.subplots(1,1)
            print(len(Merged_OOP_Rotations_Radians))
            print(len([q[j] for q in Parameter_Lists]))
            plt.scatter(Merged_OOP_Rotations_Radians,[q[j] for q in Parameter_Lists],color=cmap(color[i]),alpha=alpha)
            plt.scatter(-1*Merged_OOP_Rotations_Radians[:0:-1],[q[j] for q in Parameter_Lists][:0:-1],color=cmap(color[i]),alpha=alpha)
            Mirror_OOP_Radians = np.concatenate((-1*Merged_OOP_Rotations_Radians[:0:-1],Merged_OOP_Rotations_Radians))
            Mirror_y_coords = [q[j] for q in Parameter_Lists][:0:-1] + [p[j] for p in Parameter_Lists]
            #Mirror_weights = [1/q for q in (weights[:0:-1])] + [1/p for p in weights]
            Mirror_weights = [q*0+1 for q in (weights[:0:-1])] + [q*0+1 for p in weights]
            #fit_obj = Fit(model_dict, x=Mirror_OOP_Radians, y=Mirror_y_coords,sigma_y=Mirror_weights,absolute_sigma=False)
            fit_obj = Fit(model_dict, x=Mirror_OOP_Radians, y=Mirror_y_coords,absolute_sigma=False)
            fit_result = fit_obj.execute()
            q = np.linspace(0,math.pi,1000)
            plt.plot(q, fit_obj.model(x=q, **fit_result.params).y, color='green', ls=':')
            opls_params.append(np.array(fit_obj.model(x=q, **fit_result.params).y))
            plt.xlabel('OPLS Parameter %d' % j,size = 24)
            plt.ylabel('OOP Rotation (Degree)',size = 24)
            plt.ylim([np.amin([q[j] for q in Parameter_Lists]) - .15*(np.amax([q[j] for q in Parameter_Lists]) - np.amin([q[j] for q in Parameter_Lists])),np.amax([q[j] for q in Parameter_Lists]) + .15*(np.amax([q[j] for q in Parameter_Lists]) - np.amin([q[j] for q in Parameter_Lists]))])
            ax.tick_params(axis="x", labelsize=18)
            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(length=4,width=4)
            plt.tight_layout()
            fig.savefig('%s_%s_Improper_OPLS_Parameters_%d_Fourier_Fit' % (ring1.Name,ring2.Name,j))
            plt.close(fig)
            os.system("scp %s_%s_Improper_OPLS_Parameters_%d_Fourier_Fit.png ./Figures" % (ring1.Name,ring2.Name,j))
            os.system("rm -f %s_%s_Improper_OPLS_Parameters_%d_Fourier_Fit.png" % (ring1.Name,ring2.Name,j))
            a_params.append([fit_result.params['a%d' % z] for z in range(1,fourier_order + 1)])
            b_params.append([fit_result.params['b%d' % z] for z in range(1,fourier_order + 1)])
            a0_params.append(fit_result.params['a0'])

        Fit_Matrix = []
        p = np.linspace(0,360,1000)
        for a,b,c,d,e in zip(opls_params[0][:300],opls_params[1][:300],opls_params[2][:300],opls_params[3][:300],opls_params[4][:300]):
            Fit_List = []
            for dih in p:
                Fit_List.append(OPLS(dih,a,b,c,d,e))
            Fit_Matrix.append(Fit_List)

        print("Step 1")

        Make_Surface_Plot(p,np.linspace(0,math.pi,1000)[:300],Fit_Matrix,'%s_%s_Fit_Energies_%d' % (ring1.Name,ring2.Name,z),Title='RI-MP2 Energies',xlabel='%s %s Dihedral (degrees)' % (ring1.Name,ring2.Name),ylabel='%s %s OOP (degrees)' % (ring1.Name,ring2.Name),Tight_Layout = False)

        p = np.linspace(0,360-360/Coarse_Grid_Size[1],Coarse_Grid_Size[1])
        q = np.linspace(0,math.pi/3-math.pi/(3*Coarse_Grid_Size[0]),Coarse_Grid_Size[0])
        U = np.zeros(Coarse_Grid_Size)
        V = np.zeros(Coarse_Grid_Size)
        Coarse_Energy = np.zeros(Coarse_Grid_Size)
        for oop_num,oop in enumerate(q):
            for dih_num,dih in enumerate(p):
                U[oop_num][dih_num] = -1 * OPLS_Derivative(dih,fourier_nonfit(oop,a_params[0],b_params[0],a0_params[0],n=fourier_order),fourier_nonfit(oop,a_params[1],b_params[1],a0_params[1],n=fourier_order),fourier_nonfit(oop,a_params[2],b_params[2],a0_params[2],n=fourier_order),fourier_nonfit(oop,a_params[3],b_params[3],a0_params[3],n=fourier_order))
                V[oop_num][dih_num] = -1 * OPLS(dih,fourier_series_derivative(oop,a_params[0],b_params[0],n=fourier_order),fourier_series_derivative(oop,a_params[1],b_params[1],n=fourier_order),fourier_series_derivative(oop,a_params[2],b_params[2],n=fourier_order),fourier_series_derivative(oop,a_params[3],b_params[3],n=fourier_order),fourier_series_derivative(oop,a_params[4],b_params[4],n=fourier_order))
                Coarse_Energy[oop_num][dih_num] = 4.184 * OPLS(dih,fourier_nonfit(oop,a_params[0],b_params[0],a0_params[0],n=fourier_order),fourier_nonfit(oop,a_params[1],b_params[1],a0_params[1],n=fourier_order),fourier_nonfit(oop,a_params[2],b_params[2],a0_params[2],n=fourier_order),fourier_nonfit(oop,a_params[3],b_params[3],a0_params[3],n=fourier_order),fourier_nonfit(oop,a_params[4],b_params[4],a0_params[4],n=fourier_order))

        print("Step 2")

        p = np.linspace(-180,180,200)
        q = np.linspace(0,math.pi,200)
        Force_x = np.zeros((200,200))
        Force_y = np.zeros((200,200))
        Energy = np.zeros((200,200))
        for oop_num,oop in enumerate(q):
            for dih_num,dih in enumerate(p):
                Force_x[oop_num][dih_num] = -1 * 4.184 * OPLS_Derivative(dih,fourier_nonfit(oop,a_params[0],b_params[0],a0_params[0],n=fourier_order),fourier_nonfit(oop,a_params[1],b_params[1],a0_params[1],n=fourier_order),fourier_nonfit(oop,a_params[2],b_params[2],a0_params[2],n=fourier_order),fourier_nonfit(oop,a_params[3],b_params[3],a0_params[3],n=fourier_order))
                Force_y[oop_num][dih_num] = -1 * 4.184 * OPLS(dih,fourier_series_derivative(oop,a_params[0],b_params[0],n=fourier_order),fourier_series_derivative(oop,a_params[1],b_params[1],n=fourier_order),fourier_series_derivative(oop,a_params[2],b_params[2],n=fourier_order),fourier_series_derivative(oop,a_params[3],b_params[3],n=fourier_order),fourier_series_derivative(oop,a_params[4],b_params[4],n=fourier_order))
                Energy[oop_num][dih_num] = 4.184 * OPLS(dih,fourier_nonfit(oop,a_params[0],b_params[0],a0_params[0],n=fourier_order),fourier_nonfit(oop,a_params[1],b_params[1],a0_params[1],n=fourier_order),fourier_nonfit(oop,a_params[2],b_params[2],a0_params[2],n=fourier_order),fourier_nonfit(oop,a_params[3],b_params[3],a0_params[3],n=fourier_order),fourier_nonfit(oop,a_params[4],b_params[4],a0_params[4],n=fourier_order))

        print("Step 3")
        a_params_list.append(a_params)
        b_params_list.append(b_params)
        a0_params_list.append(a0_params)
        Fit_Energies_list.append(Energy)
        Fit_Coarse_Energies_list.append(Coarse_Energy)
        Force_x_list.append(Force_x)
        Force_y_list.append(Force_y)

    return Force_x_list,Force_y_list,Fit_Energies_list,Fit_Coarse_Energies_list,a_params_list,b_params_list,a0_params_list,opls_params,Barriers

def Make_Surface_Plot(x_axis,y_axis,Surface_Data,File_Name,Title="",xlabel="",ylabel="",Tight_Layout = True,vmin=0,vmax=10):
#This function takes in x and y-information nad surface data and prints a pretty plot from it (3-D data)
    fig,ax = plt.subplots(1,1)
    x,y = np.meshgrid(x_axis,y_axis)
    c = ax.pcolor(x,y,Surface_Data,cmap = 'seismic',vmin=vmin,vmax=vmax)
    ax.set_title(Title,fontdict = {'fontsize':24})
    plt.xlabel(xlabel,size = 24)
    plt.ylabel(ylabel,size = 24)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(length=4,width=4)
    cbar = fig.colorbar(c,ax=ax)
    cbar.ax.tick_params(labelsize = 20)
    if Tight_Layout:
        plt.tight_layout()
    fig.savefig(File_Name)

    plt.close(fig)

def Read_Input(Input_File,XYZ_File,Polymer_Name):
#This function reads in the parameter file, assigns the atoms to rings, adds available LJ, coulombic, bonded, angular, dihedral, and improper potentials, and tells the program whether it needs to parameterize missing bond potentials or partial charges. Returns Ring_List: NumPy array of Ring objects categorizing all available atoms into separate rings; Paramaterize_Bond: Boolean that equals "True" if bond parameters for interring bonds have not been specified; Paramaterize_Charges: Boolean that equals "True" if partial charges for atoms have not been specified
    Supermolecule = Molecule.Molecule(XYZ_File)
    #Open file and read lines
    f = open(Input_File)
    lines = f.readlines()
    Ring_List = []
    Ring_Read = False
    Bonded_Read = False
    Aux_Read = False
    Plumed_List = [[],[]]
    for line in lines:
        if len(line.strip().split()) > 0 and line.strip().split()[0] == "***":
            Bonded_Read = False
            for b_atoms in Bonded_Atom_List:
                Bonded_Atom_Vectors.append(Supermolecule.Get_Atom(b_atoms[1]).Position - Supermolecule.Get_Atom(b_atoms[0]).Position)
            New_Ring = Ring.Ring(Supermolecule,Ring_Atom_List,Core_Atom_List,Bonded_Atom_List,Bonded_Atom_Vectors,Ring_Name,Ring_ID,Polymer_Name,Ring_Nickname,Symmetric = Symmetry,Aux_Ring_List=Aux_Rings,Plumed_Rings=Plumed_List)
            Atom_Plumed_List = [[],[]]
            for x,ap_list in enumerate(Plumed_List):
                for atom in ap_list:
                    Atom_Plumed_List[x].append(Supermolecule.Get_Atom(atom))
            #New_Ring.Add_Plumed_Rings(Atom_Plumed_List)
            Plumed_List = [[],[]]
            Ring_List.append(New_Ring)
        if Bonded_Read:
            String_Bonded_Atom_List = line.strip().split()
            Bonded_Atom_List.append([int(i) for i in String_Bonded_Atom_List])
        if Aux_Read:
            if Aux_Count <= Aux_Ring_Num:
                Aux_Rings.append([[],[],int(line.strip().split()[1]),int(line.strip().split()[2]),line.strip().split()[0],line.strip().split()[3]])
                Aux_Count += 1
            else:
                Aux_Read = False
                Ring_Read = True
        if Ring_Read:
            if len(line.strip().split()) == 1 or "CORE" in line.strip().split()[-1]:
                Ring_Atom_List.append(int(line.strip().split()[0]))
                if "CORE" in line.strip().split()[-1]:
                    Core_Atom_List.append(int(line.strip().split()[0]))
                    if line.strip().split()[-1] == "CORE1" or line.strip().split()[-2] == "CORE1":
                        Plumed_List[0].append(int(line.strip().split()[0]))
                    if line.strip().split()[-1] == "CORE2":
                        Plumed_List[1].append(int(line.strip().split()[0]))
            elif line.strip().split()[-1][:3] == "AUX":
                Aux_Rings[int(line.strip().split()[-1][-1])][0].append(int(line.strip().split()[0]))
                if len(line.strip().split()[-1]) > 5:
                    Aux_Rings[int(line.strip().split()[-1][-1])][1].append(int(line.strip().split()[0]))
            elif line.strip().split()[0] == "Bonded":
                Ring_Read = False
                Bonded_Read = True
        if len(line.strip().split()) > 0 and line.strip().split()[0] == "Ring":
            Ring_ID = int(line.strip().split()[1])
            if line.strip().split()[2].strip() == "True":
                Symmetry = True
            else:
                Symmetry = False
            Ring_Name = line.strip().split()[3].strip()
            Ring_Nickname = line.strip().split()[4].strip()
            Aux_Ring_Num = int(line.strip().split()[5].strip())
            if Aux_Ring_Num != 0:
                Aux_Read = True
                Aux_Count = 1
            else:
                Ring_Read = True
            Ring_Atom_List = []
            Bonded_Atom_List = []
            Bonded_Atom_Vectors = []
            Core_Atom_List = []
            Aux_Rings = []
            #if Aux_Ring_Num != 0:

    return Ring_List,False,False

Polymers = ["P3HT","PTB7","PTB7","PNDI_T"]
Input_Files = ["P3HT_Input.txt","PTB7_Input.txt","PTB7_Input.txt","PNDI_T_Input.txt"]
XYZ_Files = ["P3HT_Input.xyz","PTB7_Input.xyz","PTB7_Input.xyz","PNDI_T_Input.xyz"]
Improper = [0,5,10,15,20,25,30]
fig, ax = plt.subplots()
counter = 0
OPLS_Parameters = [np.array([0.4540,3.0589,-0.1075,-0.1850]),np.array([-0.0224,4.0413,-0.1474,-0.2611]),np.array([0.1902,3.9948,-0.4347,-0.6789]),np.array([0.4270,5.7845,0.3723,-0.0068])]

for pol_name,opls_params,Input_File,XYZ_File in zip(Polymers,OPLS_Parameters,Input_Files,XYZ_Files):
    Polymer_Name = Input_File.split('.')[0]
    Ring_List,Parameterize_Bond,Parameterize_Charges = Read_Input(Input_File,XYZ_File,Polymer_Name)
    Barriers = []
    Improper_Energies = []
    Deloc_Energies = []
    if pol_name != "PTB7":
        #f = open("./Permanent_Outputs/%s_Input/Delocalization_Energies_Choice.txt" % pol_name,'r')
        f = open("./Permanent_Outputs/%s_Input/Delocalization_Energies.txt" % pol_name,'r')
    else:
        #f = open("./Permanent_Outputs/%s_Input/Delocalization_Energies_Choice_%d.txt" % (pol_name,counter),'r')
        f = open("./Permanent_Outputs/%s_Input/Delocalization_Energies_%d.txt" % (pol_name,counter),'r')

    lines = f.readlines()
    for i,line in enumerate(lines[:7]):
        String_Energies = line.strip().split()
        Energies = [float(energy) * 4.184 for energy in String_Energies]
        if pol_name == "PTB7" and counter == 0 and i == 0:
            Barriers.append(Energies[27]-Energies[0])
        else:
            Barriers.append(Energies[9]-Energies[0])
        if pol_name == "PTB7" and counter == 0 and i == 0:
            Improper_Energies.append(Energies[18])
        else:
            Improper_Energies.append(Energies[0])
        Deloc_Energies.append(Energies)

    #added 4/22
    """Barriers = []
    Rotated_Shape = (7,36)
    Merged_OOP_Rotations_Degrees = [0.0,5.0,10.0,15.0,20.0,25.0,30.0]
    print([np.ones(np.array(Deloc_Energies).shape)])
    print([Deloc_Energies])
    Force_x,Force_y,Fit_Energies,Deloc_Coarse_Energies,a_params_list,b_params_list,a0_params_list,opls_params,Barriers = Fit_Improper_Energies([Deloc_Energies],Ring_List,Rotated_Shape,Merged_OOP_Rotations_Degrees,[np.ones(np.array(Deloc_Energies).shape)])
    """
    Improper_Energies = np.array(Improper_Energies) - Improper_Energies[0]
    E_degrees = []

    line = lines[0]
    String_Energies_No_Improper = line.strip().split()
    Energies_No_Improper = [float(energy) * 4.184 for energy in String_Energies]

    Equivalent_Dih = []

    print(Energies_No_Improper)

    for i_energy in Improper_Energies:
        for z,dih_energy in enumerate(Energies_No_Improper):
            Total_dih_energy = dih_energy - Energies_No_Improper[0]
            if Total_dih_energy > i_energy and z != 0:
                linear_slope = (Energies_No_Improper[z] - Energies_No_Improper[z-1])/10
                y_int = Energies_No_Improper[z-1] - ((z-1) * 10 * linear_slope)
                print(Energies_No_Improper[z-1])
                print(z-1)
                print(linear_slope)
                equivalent_degrees = (i_energy - y_int)/linear_slope
                print(equivalent_degrees)
                E_degrees.append(equivalent_degrees)
                break

    Improper_Energy = []
    Dihedral_Energy = []
    for angle,energy_row in zip(Improper,Energies):
        angle_radians = math.radians(angle)
        Dihedral_Energy.append(.5*4.184*(opls_params[0]*(1+math.cos(angle_radians)) +opls_params[1]*(1-math.cos(2*angle_radians)) + opls_params[2]*(1+math.cos(3*angle_radians)) + opls_params[3]*(1-math.cos(4*angle_radians))))
    if counter == 0 and pol_name == "PTB7":
        pol_name = pol_name + " F-out"
    elif counter == 1 and pol_name == "PTB7":
        pol_name = pol_name + " F-in"
    elif pol_name == "PNDI_T":
        pol_name = "PNDI-T"
    """
    plt.plot(Improper,Barriers,marker='s',linewidth=5,markeredgecolor='k',markersize=8,label=pol_name)
    plt.xlabel("Improper Angle (degrees)",size = 24)
    plt.ylabel("Torsional Barrier (kJ/mol)",size = 24)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(length=4,width=4)

    plt.legend(fontsize="x-large")
    plt.tight_layout()
    if pol_name != "PTB7 F-out" and pol_name != "PTB7 F-in":
        fig.savefig("%s_Barrier_With_Improper" % pol_name)
    else:
        fig.savefig("%s_Barrier_With_Improper_%d" % (pol_name,counter))
        counter += 1
    print(Improper_Energies)
    """
    Dihedral_Energy = np.array(Dihedral_Energy)
    Dihedral_Energy = Dihedral_Energy - Dihedral_Energy[0]
    
    plt.plot(Improper,Improper_Energies,color='b',marker='s',linewidth=5,markersize=8,label="Improper Torsion")
    plt.plot(Improper,Dihedral_Energy,color='r',marker='s',linewidth=5,markersize=8,label="Dihedral Torsion")
    plt.xlabel("Angle (degrees)",size = 24)
    plt.ylabel("Energy (kJ/mol)",size = 24)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(length=4,width=4)
    plt.tight_layout()
    if counter == 0 and pol_name == "PTB7 F-out":
        counter += 1
    plt.legend(loc="upper left",fontsize="x-large")
    fig.savefig("%s_Energy_Comparison" % pol_name)
    plt.close()
    fig, ax = plt.subplots()

plt.close()