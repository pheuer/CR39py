# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:23:42 2023

@author: pheu
"""

from pathlib import Path

from cr39py.scan import Scan
from cr39py.cut import Cut

cpsa_path = Path("C:\\", "Users", "pheu", "Data", "data_dir", "105350", "o105350-Ernie-PR3236-2hr_40x_s0.cpsa")

scan = Scan.from_cpsa(cpsa_path)


# First we'll plot all of the tracks
scan.cutplot()


# Now we'll add some cuts to exclude tracks that are more likely to be noise

# Tracks with very high contrast are likely to be noise
scan.add_cut(Cut(cmin=40))

# Tracks with large diameters are also likely to be noise 
scan.add_cut(Cut(dmin=12))

# Now we plot again, and the image is much clear with all the noise removed!
scan.cutplot()