# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:36:27 2023

@author: pheu
"""

from pathlib import Path

from cr39py.scan import Scan
cpsa_path = Path("C:\\", "Users", "pheu", "Data", "data_dir", "105350", "o105350-Ernie-PR3236-2hr_40x_s0.cpsa")

scan = Scan.from_cpsa(cpsa_path)

# Launch the command line interface for making cuts
scan.cli()