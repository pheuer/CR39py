# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:51:31 2022

@author: pvheu
"""
import os
import numpy as np
import astropy.units as u
from cr39py import root_dir

from collections import namedtuple


# namedtuple for SRIM data
SRIMdata= namedtuple('SRIMdata', ['ion_energy',
                                       'dEdx_electronic',
                                       'dEdx_nuclear',
                                       'projected_range',
                                       'longitudinal_straggling',
                                       'lateral_straggling'])

def read_srim(file):
    """
    Reads a SRIM output file and returns the values within as a
    
    file : str
        Filepath to a SRIM output file 
        
    Returns
    -------
    
    output : namedtuple, 'SRIMdata'
        A namedtuple of u.Quantity arrays with the following contents
        
        ion_energy -> ion energy in eV
        dEdx_electronic -> electronic stopping power in keV/um
        dEdx_nuclear -> nuclear stopping power in keV/um
        projected_range -> projected range in m
        longitudinal_straggling -> longitudial straggling in m
        lateral_straggling -> lateral straggling in m
    
    
    
    """
    
    
    
    ion_energy = []
    energy_convert = {'eV':1, 'keV':1e3, 'MeV':1e6, 'GeV':1e9}
    range_convert = {'A':1e-10, 'nm':1e-9, 'um':1e-6, 'mm':1e-3, 'cm':1e-2}
    dEdx_electronic = []
    dEdx_nuclear = []
    projected_range = []
    longitudinal_straggling = []
    lateral_straggling = []
    
    with open(file, 'r') as f:
        # Read in the file contents
        c = f.readlines()
        
    # Skip header
    c = c[23:]
    
    while True:
        s = c[0].split(' ')
        # Discard any single space strings
        s = [st for st in s if st != '']
        
        # The first column is the ion energy, along with a unit
        energy, unit = s[0], s[1]
        if unit not in energy_convert.keys():
            raise ValueError(f"Unrecognized energy unit: {unit}")
        ion_energy.append(float(energy)*energy_convert[unit])
        
        
        # Read the dEdx electronic and nuclear
        # These get read in as a pair, along with another awkward empty string
        # because of the spacing...
        electronic, nuclear = s[2], s[3]
        dEdx_electronic.append(float(electronic))
        dEdx_nuclear.append(float(nuclear))
        
        # Read the stopping powers
        rng, unit = s[4], s[5]
        if unit not in range_convert.keys():
            raise ValueError(f"Unrecognized range unit: {unit}")
        projected_range.append(float(rng)*range_convert[unit])
        
        rng, unit = s[6], s[7]
        if unit not in range_convert.keys():
            raise ValueError(f"Unrecognized range unit: {unit}")
        longitudinal_straggling.append(float(rng)*range_convert[unit])
        
        rng, unit = s[8], s[9]
        if unit not in range_convert.keys():
            raise ValueError(f"Unrecognized range unit: {unit}")
        lateral_straggling.append(float(rng)*range_convert[unit])

        # If the next line contains the dotted line at the end of the file,
        # terminate the loop
        if '--' in c[1]:
            break
        # Else remove the line we just read and start again
        else:
            c = c[1:]
                
        
    ion_energy = np.array(ion_energy)*u.eV
    dEdx_electronic = np.array(dEdx_electronic) * u.keV/u.um
    dEdx_nuclear = np.array(dEdx_nuclear) * u.keV/u.um
    projected_range = np.array(projected_range)*u.m
    longitudinal_straggling = np.array(longitudinal_straggling)*u.m
    lateral_straggling = np.array(lateral_straggling )*u.m
    
    output = SRIMdata(ion_energy=ion_energy, dEdx_electronic=dEdx_electronic,
                      dEdx_nuclear=dEdx_nuclear, projected_range=projected_range,
                      longitudinal_straggling=longitudinal_straggling,
                      lateral_straggling=lateral_straggling)
    
    return output
    
    

if __name__ == '__main__':
    path = os.path.join(root_dir, 'data', 'Deuterium in Tungsten.txt')
    output = read_srim(path)
    print(output.ion_energy)