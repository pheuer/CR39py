# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:19:29 2023

@author: pheu
"""

from pathlib import Path

from cr39py.scan import Scan
cpsa_path = Path("C:\\", "Users", "pheu", "Data", "data_dir", "105350", "o105350-Ernie-PR3236-2hr_40x_s0.cpsa")

scan = Scan.from_cpsa(cpsa_path)


# Make a plot of the cr39
scan.plot()


# Make a plot of the Z-focus plane (useful for seeing if a scan has gone wrong)
scan.focus_plot()

# Make a plot with more information about the tracks
scan.cutplot()