# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Adapted in part from code written by Hans Rinderknecht
"""
import os
import h5py
import numpy as np

from collections import namedtuple

from fast_histogram import histogram2d
import matplotlib.pyplot as plt

from cr39py.util.misc import find_file
from cr39py.util.cli import _cli_input
from cr39py.cuts import Cut, Subset


# Represents metadata from a single frame of cr39
# used when reading CPSA files
FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])




def energy_from_diameter(diameter, etchtime, particle='D',
                         V_bulk=None, 
                         k=None, n=None):
    """
    Estimate the energy of particle that created a particle track based on
    its diameter 
    
    
    Parameters
    ----------
    
    diameter : np.ndarray
        Track diameters in um
        
    etchtime : float
        Etch time in hours
        
    particle : str
        One of: 
            'P' -> proton
            'D' -> deuteron
            'T' -> triton
        
        Default is 'D'
        
    V_bulk : float
        Emperical constant. Default value is 2.66 um/hr
        
    k : float
        Emperical constant. Default value is the mean of the data in Table
        III : k = 0.782
        
    n : float
        Emperical constant. Default value is the mean of the data in Table
        III : n = 1.241
        
        
    Returns
    -------
    
    E : np.ndarray
        Energy in MeV
    
    
    Notes
    -----
    Lahmann, et al., RSI 91, 053502 (2020)
    2-parameter model for CR-39 D(E): 
     D(E) = 2*time_hr*V_bulk/(1+k*(E_MeV/Z^2/A)^n)  (Eq. 5)
     
    Which, inverted for E_MeV, gives
    
    E_MeV = Z^2 A [ (2*time_hr*V_bulk/D -1)/k]^(1/n)
    
    Typical values for (k, n) actually vary with etch time...
    From Table III, for CPS2 deuteron data:
       time    k       n 
       2.0 hr  0.868   1.322
       3.0 hr  0.809   1.103
       3.0 hr  0.781   1.198
       3.0 hr  0.671   1.340
    Given all this, about the best we can do is take typical
    values and know that a given piece will have variation.
    From simulations (matched to typical values)
       time    k       n
       1 hr    0.894   1.343
       3 hr    0.829   1.413
       5 hr    0.769   1.467
    
    """
    
    if V_bulk is None:
        V_bulk = 2.66
        
    if k is None:
        k = 0.782
        
    if n is None:
        n = 1.241
        
    if particle.lower() == 'p':
        Z=1
        A=1
    elif particle.lower() == 'd':
        Z=1
        A=2
    elif particle.lower() == 't':
        Z=1
        A=3
    else:
        raise ValueError("Particle must be one of ['D', 'T'], but provided "
                         f"value was {particle}")
        
    energy = Z**2*A*((2*etchtime*V_bulk/diameter - 1)/k)**(1/n)
    
    return energy



                        

class CR39:

    # Axes dictionary for trackdata
    axes_ind = {'X':0, 'Y':1, 'D':2, 'C':3, 'E':5, 'Z':6}
    ind_axes = ['X', 'Y', 'D', 'C', 'E', 'Z']
    
    def __init__(self, *args, verbose=False, data_dir=None, subsets=None):
        
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
            
        subsets : list of Subset objects
            A list of subsets to attach to this piece
            
        """
        self.verbose = verbose
        
        # Initialize the subsets list
        self.subsets = []
        # If subests is None, initialze with a single blank subset
        if subsets is None:
            self.subsets = [Subset(),]
        # If subsets is a h5py.Grou or a string, try to load from there
        elif isinstance(subsets, (h5py.Group, str)):
            self.load(subsets)
            
        # If subsets is a list, assume it is a list of subsets
        elif isinstance(subsets, list):
            self.subsets = subsets
        else:
            raise ValueError("Invalid type for kwarg 'subsets'")
        
        # The index of the currently selected subset
        self.current_subset_i = 0
        
        # Store figures once created for blitting
        self.plotfig = None
        self.cutplotfig = None
        

        # If the first argument is a file path, load the file directly
        # if not, assume it is a shot number and look for it 
        if isinstance(args[0], str):
            self.path = args[0]
        else:
            if data_dir is None:
                raise ValueError("The 'data_dir' keyword is required in order "
                                 "to locate a file based on a shot number.")
            
            self.path = self._find_data(args[0], data_dir)
        
        self._read_CPSA(self.path)
        
        
    def save(self, grp):
           """
           Save the data about this dataset into an h5 group
           
           grp : h5py.Group or path string
               The location to save the h5 data. This could be the root group
               of its own h5 file, or a group within a larger h5 file.
               
               If a string is provided instead of a group, try to open it as
               an h5 file and create the group there
           
           """
           if isinstance(grp, str):
               with h5py.File(grp, 'w') as f:
                   self._save(f)
           else:
               self._save(grp)
               
    def _save(self, grp):
           """
           See docstring for "save"
           """
           grp.attrs['path'] = self.path
           grp.attrs['current_subset_i'] = self.current_subset_i
           
           subsets_grp = grp.create_group('subsets')
           for i, subset in enumerate(self.subsets):
               subset_grp = subsets_grp.create_group(f"subset_{i}")
               subset.save(subset_grp)

       
    def load(self, grp):
           """
           Load this dataset from an h5 group
           
           grp : h5py.Group 
               The location from which to load the h5 data. This could be the 
               root group of its own h5 file, or a group within a larger h5 file.
           
           """
           # Initialize the subsets list as empty
           self.subsets = []
           
           
           if isinstance(grp, str):
               with h5py.File(grp, 'r') as f:
                   self._load(f)
           else:
               self._load(grp)
               
           self.apply_cuts()
           self.cutplot()
              
                        
    def _load(self, grp):
        """
        See documentation for 'load'
        """
        self.path = str(grp.attrs['path'])
        self.current_subset_i = int(grp.attrs['current_subset_i'])
        
        # Load the cuts
        subsets_grp = grp["subsets"]
        for key in subsets_grp:
            self.subsets.append(Subset(subsets_grp[key]))
            
        
        
    def _log(self, msg):
        if self.verbose:
            print(msg)
            
            
    @property
    def current_subset(self):
        return self.subsets[self.current_subset_i]
            
            
    def _find_data(self, id, data_dir):
        self.data_dir = data_dir
        self.file_dir = os.path.join(self.data_dir, str(id))
            
        # Verify the data_dir exists
        if not os.path.isdir(self.file_dir):
            raise ValueError(f"The file directory {self.file_dir} does not exist.")
        
        # Find the file path
        path = find_file(self.file_dir, [ '.cpsa'])
        
        return path
        
        
    def _read_CPSA(self, path):

        with open(path, 'rb') as file:

            # First read in the header values
            self._log("Reading CPSA header")
            self.version = -np.fromfile(file, count=1, dtype='int32' )[0]
            self._log(f"Version: {self.version}")
            
    
            # Number of microscope frames in the x and y directions of the scan
            # respectively
            self.nx = np.fromfile(file, count=1, dtype='int32' )[0]
            self.ny = np.fromfile(file, count=1, dtype='int32' )[0]
            self.nframes = self.nx*self.ny
            self._log(f"nx, ny microscope bins: {self.nx}, {self.ny}")
            self._log(f"Nframes: {self.nframes}")
            
            # h[3] is "Nbins" which is not used except with legacy data\
            Nbins = np.fromfile(file, count=1, dtype='int32' )[0]
    
            # Pixel size in microns: note that this is stored as a single
            # so needs to be read as such
            self.pix_size = np.fromfile(file, count=1, dtype='single' )[0]
            self._log(f"Pixel size: {self.pix_size:.1e} um")
            
            # h[5] is "ppb" which is not used except with legacy data
            pbb = np.fromfile(file, count=1, dtype='single' )[0]
            
            # Thresholds for border, contrast, eccentricity, 
            # and number of eccentricity moments
            self.threshold =np.fromfile(file, count=4, dtype='int32' )[0]
            self._log(f"Threshold: {self.threshold}")
            
            # Number of utilized camera image pixels in the x and y directions
            self.NFPx = np.fromfile(file, count=1, dtype='int32' )[0]
            self.NFPy = np.fromfile(file, count=1, dtype='int32' )[0]
            self._log(f"Untilized camera image px NFPx, NFPy: {self.NFPx}, {self.NFPy}")
            
            # Microscope frame size in microns
            self.fx = self.pix_size*self.NFPx
            self.fy = self.pix_size*self.NFPy
            self._log(f"Microscope frame size fx, fy: {self.fx:.1e} um, {self.fy:.1e} um")
        
            # Read the full datafile as int32 and separate out the track info
            self._log("Reading CPSA data")
            
            # Frame headers stores the info from each frame header
            self.frame_headers = []
            # Tracks in each frame
            tracks = []
            # Keep track of the total number of hits (tot_hits) and the
            # running total of hits (cum_hits) which will be used later
            # for assembling the trackdata array
            tot_hits = 0
            cum_hits = np.array([0], dtype='int32')
            
            
            # Collect x and y positions of frames in sets that, once sorted
            # will be the x and y axes of the dataset
            self.xax = np.zeros(self.nx)
            self.yax = np.zeros(self.ny)
            for i in range(self.nframes):
                if i % 5000 == 4999:
                    self._log(f"Reading frame {i+1}/{self.nframes}")
                
                # Read the bin header
                h = np.fromfile(file, count=10, dtype='int32' )
                
                # Header contents are as follows
                # 0 -> frame number (starts at 1)
                # 1 -> xpos (frame center x position, in 1e-7 m)
                # 2 -> ypos (frame center y position, in 1e-7 m)
                # 3 -> hits (number of tracks in this frame)
                # 4,5,6 -> BLENR (something about rejected tracks?)
                # 7 -> zpos (microscope focus? Units? )
                # 8 -> x_ind (x index of the frame, staring at 0)
                # 9 -> y_ind (y index of the frame, staring at 0)

                fh = FrameHeader(number=h[0],
                                 xpos=h[1], ypos=h[2], 
                                 hits=h[3],
                                 BLENR=h[4:7], 
                                 zpos=h[7], 
                                 x_ind=h[8], y_ind = h[9])
                self.frame_headers.append(fh)
                
                # Put the bin x and y values in the appropriate place in the axes
                self.xax[fh.x_ind] = fh.xpos*1e-5
                self.yax[fh.y_ind] = fh.ypos*1e-5
   
                # Increment the counters for the number of hits
                tot_hits += fh.hits
                cum_hits = np.append(cum_hits, tot_hits)
                
                # Read the track data for this frame
                # Each frame entry contains a sequence for each hit, which
                # contains the following integers
                # 1) diameter (int16)  in units of 1e-2*pix_size (?)
                # 2) ecentricity (uint) 
                # 3) contrast (uint)
                # 4) avg contrast (uint)
                # 5) x pos (int16) in units of 1e-4*pix_size
                # 6) y pos (int16) in units of 1e-4*pix_size
                # 7) z pos (int16) in units of 1e-2*pix_size (??)
                #
                # The x and y pos are relative to the upper right corner 
                # of the current frame
                
                t= np.zeros([fh.hits, 7])
                if fh.hits > 0:
                    
                    # Diameters
                    t[:, 2] = np.fromfile(file, count=fh.hits, 
                                          dtype='int16')*1e-2*self.pix_size
                    
                    # Ecentricities
                    t[:, 5] = np.fromfile(file, count=fh.hits, dtype='byte')
                    
                    # Contrast
                    t[:, 3] = np.fromfile(file, count=fh.hits, dtype='byte')
                    
                    # Avg Contrast
                    t[:, 4] = np.fromfile(file, count=fh.hits, dtype='byte')
                    
                    
                    # x position, cm
                    # Positions are relative to the top right of the current
                    # frame, so we need to adjust them accordingly
                    t[:, 0] = (- np.fromfile(file, count=fh.hits, 
                                          dtype='int16')*self.pix_size*1e-4
                               + fh.xpos*1e-5 
                               + (self.fx/2)*1e-4)

                     # y position, cm
                    t[:, 1] = ( np.fromfile(file, count=fh.hits, 
                                          dtype='int16')*self.pix_size*1e-4
                               + fh.ypos*1e-5
                               - (self.fy/2)*1e-4 )
                    
                    # z position, microns
                    t[:, 6] = fh.zpos*self.pix_size*1e-2
    
                    
                tracks.append(t)
        self._log("Done Reading CPSA data")
    
        
        self._log("Processing the tracks")
        

        # The order of the quantities in track data is: 
        # 0) x position
        # 1) y position
        # 2) diameter
        # 3) contrast
        # 4) avg contrast
        # 5) ecentricity
    
        # Re-shape the track data into a list of every track
        self.trackdata = np.zeros([tot_hits, 7])
        for i in range(self.nframes):
            self.trackdata[cum_hits[i]:cum_hits[i+1], :] = tracks[i]
            
        # ntracks will change to keep track of the number currently selected
        self.ntracks = tot_hits
        # raw_ntracks will always record the number on the entire piece
        self.raw_ntracks = tot_hits
        
        # Sort the tracks by diameter for later slicing into energy dslices
        isort = np.argsort(self.trackdata[:,2])
        self.trackdata = self.trackdata[isort, :]
        
        # Store all the tracks, as a starting point for making cuts on
        # self.trackdata
        self.raw_trackdata = np.copy(self.trackdata)
        
        # Sort the yaxis (it's backwards...)
        self.yax = np.sort(self.yax)
        
        # Make axes for the other quantites
        self.dax = np.linspace(0, 20, num=40)
        self.cax = np.linspace(0, 80, num=80)
        self.eax = np.linspace(0, 50, num=60)
        
        

    def frames(self, axes=('X', 'Y'), trim=True, hax=None, vax=None):
        """
        Create a histogram of the track data
        
        axes : tuple of 2 or 3 str
            The first two values represent the axes of the histogram. If no
            third value is included, then the resulting histogram is of the 
            number of hits in each  bin. If a third value is included,
            the histogram will be of that value in each bin
            
            Chose from the following: 
            'X': x position
            'Y': y position
            'D': diameter
            'C': contrast
            'E': ecentricity 
            
            The default is ('X', 'Y')
            
        trim : bool
            If true, trim the array and axes down to exclude any entirely 
            empty rows or columns
            
            
        hax, vax : np.ndarrays
            If set, replaces the default axes 
        
        """

        
        axdict = {'X':self.xax, 'Y':self.yax, 'D':self.dax,
                'C':self.cax, 'E':self.eax}
        
        i0 = self.axes_ind[axes[0]]
        i1 = self.axes_ind[axes[1]]
        
        if hax is None:
            ax0 = axdict[axes[0]]
        else:
            ax0 = hax
            
        if vax is None:
            ax1 = axdict[axes[1]]
        else:
            ax1 = vax
        
        # If creating a histogram like the X,Y,D plots
        if len(axes) == 3:
            i2 = self.axes_ind[axes[2]]
            weights = self.trackdata[:, i2]
        else:
            weights = None
            
            
        # If the implied range is < 25x the spacing of the axis, 
        # create a higher resolution axis that spans this range
        dax0=np.mean(np.gradient(ax0))
        if (np.max(ax0) - np.min(ax0)) < 10*dax0:
            ax0 = np.linspace(np.min(ax0), np.max(ax0), num=30)
        dax1=np.mean(np.gradient(ax1))
        if (np.max(ax1) - np.min(ax1)) < 10*dax1:
            ax1 = np.linspace(np.min(ax1), np.max(ax1), num=30)
        
        
        
        rng = [(np.min(ax0), np.max(ax0)), (np.min(ax1), np.max(ax1))]
        bins = [ax0.size, ax1.size]

        xax = np.linspace(rng[0][0], rng[0][1], num=bins[0])
        yax = np.linspace(rng[1][0], rng[1][1], num=bins[1])
        arr = histogram2d(self.trackdata[:,i0],self.trackdata[:,i1],
                          bins=bins, range=rng, weights=weights)
        

        # Create the unweighted histogram and divide by it (sans zeros)
        if len(axes)==3:
            arr_uw = histogram2d(self.trackdata[:,i0],
                                    self.trackdata[:,i1],
                                    bins=bins, range=rng)
            nz = np.nonzero(arr_uw)
            arr[nz] = arr[nz]/arr_uw[nz]
            
            
        # Get rid of any entirely zero rows or columns
        if trim:
            xi = np.nonzero(np.sum(arr, axis=1))[0]
            if len(xi)==0:
                xa, xb = 0, -1
            else:
                xa, xb = xi[0], xi[-1]+1
    
            yi = np.nonzero(np.sum(arr, axis=0))[0]
            if len(yi)==0:
                ya, yb = 0, -1
            else:
                ya, yb = yi[0], yi[-1]+1
            
            xax = xax[xa:xb]
            yax = yax[ya:yb]
            arr = arr[xa:xb, ya:yb]
        
        return xax, yax, arr
    
    
    def hreflect(self):
        self.raw_trackdata[:, 0] *= -1
        self.trackdata[:, 0] *= -1
        self.plot()
        
    def vreflect(self):
        self.raw_trackdata[:, 1] *= -1
        self.trackdata[:, 1] *= -1
        self.plot()
        
        
    # ***********************************************************
    # Cut functions
    # These all apply to the current subset and are just wrappers
    # for the matching method on that class 
    # ***********************************************************
    def add_cut(self, c):
         self.current_subset.add_cut(c)
            
    def remove_cut(self, i):
        self.current_subset.remove_cut(i)

    def replace_cut(self, i, c):
         self.current_subset.replace_cut(i,c)
           
    # ***********************************************************
    # Subset functions
    # ***********************************************************
    def select_subset(self, i):
        if i > len(self.subsets)-1:
            print(f"Cannot select the {i} subset, there are only "
                             f"{len(self.subsets)} subsets.")
        else:
            self.current_subset_i = i
        
    def add_subset(self, *args):
        """
        Add a new subset
        
        If no arguement is given create an empty subset
        
        Otherwise, assume the argument is a subset
        """
        
        if len(args)==1:
            s = args[0]
        else:
            s= Subset()
        self.subsets.append(s)
        
    def remove_subset(self, i):
        if i > len(self.subsets)-1:
            print(f"Cannot remove the {i} subset, there are only "
                             f"{len(self.subsets)} subsets.")
        else:
            self.subsets.pop(i)

    def replace_subset(self, i, sc):
         self.subsets[i] = sc
        
        
    def apply_cuts(self, use_cuts=None, invert=False):
        """
        Apply cuts to the track data
        
        use_cuts : int, list of ints (optional)
            If provided, only the cuts corresponding to the int or ints 
            provided will be applied. The default is to apply all cuts
            
        invert : bool (optional)
            If true, return the inverse of the cuts selected. Default is 
            false.
            

        """

        valid_cuts = list(np.arange(len(self.current_subset.cuts)))
        if use_cuts is None:
            use_cuts = valid_cuts
        else:
            for s in use_cuts:
                if s not in valid_cuts:
                    raise ValueError(f"Specified cut index is invalid: {s}")
        use_cuts = list(use_cuts)    

        keep = np.ones(self.raw_ntracks).astype(bool)   
        
        for i, cut in enumerate(self.current_subset.cuts):
            if i in use_cuts:
                # Get a boolean array of tracks that are inside this cut
                x = cut.test(self.raw_trackdata)
                
                # negate to get a list of tracks that are NOT
                # in the excluded region (unless we are inverting)
                if not invert:
                    x = np.logical_not(x)
                keep *= x
                
        # Regardless of anything else, only show tracks that are within
        # the domain
        if self.current_subset.domain is not None:
            keep *= self.current_subset.domain.test(self.raw_trackdata)
            
        # Select only these tracks
        self.trackdata = self.raw_trackdata[keep, :]
        
        
        # Calculate the bin edges for each dslice
        # !! note that the tracks are already sorted into order by diameter
        # when the CR39 data is read in
        if self.current_subset.current_dslice is not None:
            # Figure out the dslice width
            dbin = int(self.trackdata.shape[0]/self.current_subset.ndslices)
            # Extract the appropriate portion of the tracks
            b0 = self.current_subset.current_dslice*dbin
            b1 = b0 + dbin
            self.trackdata = self.trackdata[b0:b1, :]
            
        self.ntracks = self.trackdata.shape[0]

        
        
    
    def cli(self):
        self.apply_cuts()
        self.cutplot()
        
        while True:
            
            print ("*********************************************************")
            print(f"Current subset index: {self.current_subset_i} of {np.arange(len(self.subsets))}")
            # Print a summary of the current subset
            print(self.current_subset)
            print(f"ntracks selected: {self.ntracks:.1e} "
                  f"(of {self.raw_ntracks:.1e})")
            
            print("add (a), edit (e), edit the domain (d), remove (r), plot (p), "
                  "plot inverse (pi), switch subsets (subset), change dslices (dslice), "
                  "change the number of dslices (ndslices), end (end), help (help)")
            split = _cli_input(mode='alpha-integer list')
            x = split[0]
            
            if x == 'help':
                print("Enter commands, followed by any additional arugments "
                      "separated by commas.\n"
                      " ** Commands ** \n"
                      "'a' -> create a new cut\n"
                      "'c' -> Select a new dslice\n"
                      "Argument (one int) is the index of the dslice to select"
                      "Enter 'all' to select all"
                      "'d' -> edit the domain\n"
                      "'e' -> edit a cut\n"
                      "Argument (one int) is the cut to edit\n"
                      "'ndslices' -> Change the number of dslices on this subset."
                      "'p' -> plot the image with current cuts\n"
                      "Arguments are numbers of cuts to include in plot\n"
                      "The default is to include all of the current cuts\n"
                      "'pi' -> plot the image with INVERSE of the cuts\n"
                      "Arguments are numbers of cuts to include in plot\n"
                      "The default is to include all of the current cuts\n"
                      "'r' -> remove an existing cut\n"
                      "Arguments are numbers of cuts to remove\n"
                      "'subset' -> switch subsets or create a new subset\n"
                      "Argument is the index of the subset to switch to, or"
                      "'new' to create a new subset"
                      "'help' -> print this documentation\n"
                      "'end' -> accept the current values\n"

                      " ** Cut keywords ** \n"
                      "xmin, xmax, ymin, ymax, dmin, dmax, cmin, cmax, emin, emax\n"
                      "e.g. 'xmin:0,xmax:5,dmax=15'\n"
                      )
                
            elif x == 'end':
                self.apply_cuts()
                self.cutplot()
                break
            
            elif x == 'a':
                print("Enter new cut parameters as key:value pairs separated by commas")
                kwargs = _cli_input(mode='key:value list')
                
                #validate the keys are all valid dictionary keys
                valid=True
                for key in kwargs.keys():
                    if key not in list(Cut.defaults.keys()):
                        print(f"Unrecognized key: {key}")
                        valid=False
                        
                if valid:
                    c = Cut(**kwargs)
                    if x == 'r':
                        ind = int(split[1])
                        self.current_subset.replace_cut(ind, c)
                    else:
                       self.current_subset.add_cut(c)
                        
                self.apply_cuts()
                self.cutplot()
                
            elif x == 'dslice':
                if len(split)<2:
                    print("Select the index of the dslice to switch to, or"
                          "enter 'all' to select all dslices")
                    ind = _cli_input(mode='alpha-integer')
                else:
                    ind = split[1]
                    
                if ind == 'all':
                    self.current_subset.set_current_dslice(None)
                else:
                    self.current_subset.set_current_dslice(int(ind))
                self.apply_cuts()
                self.cutplot()
                    
                    
                
            elif x == 'd':
                print("Current domain: " + str(self.current_subset.domain))
                print("Enter a list key:value pairs with which to modify the domain"
                      "(set a key to 'None' to remove it)")
                kwargs = _cli_input(mode='key:value list')
                for key in kwargs.keys():
                    if str(kwargs[key]).lower() == 'none':
                        self.current_subset.domain.dict[key] = None
                    else:
                        self.current_subset.domain.dict[key] = float(kwargs[key])
                self.apply_cuts()
                self.cutplot()
 
                        
            elif x == 'e':
                if len(split)>1:
                    ind = int(split[1])
                    
                    print(f"Selected cut ({ind}) : " + str(self.current_subset.cuts[ind]))
                    print("Enter a list key:value pairs with which to modify this cut"
                          "(set a key to 'None' to remove it)")
                    
                    kwargs = _cli_input(mode='key:value list')
                    for key in kwargs.keys():
                        if str(kwargs[key]).lower() == 'none':
                            self.current_subset.cuts[ind].dict[key] = None
                        else:
                            self.current_subset.cuts[ind].dict[key] = float(kwargs[key])
                            
                    self.apply_cuts()
                    self.cutplot()
                else:
                    print("Specify the number of the cut you want to modify "
                          "as an argument after the command.")
                        
            elif x == 'ndslices':
                if len(split)<2:
                    print("Enter the requested number of dslices")
                    ind = _cli_input(mode='alpha-integer')
                else:
                    ind = split[1]
                    
                self.current_subset.set_ndslices(int(ind))

                    
            elif x in ['p', 'pi']:
                if x =='pi':
                    invert=True
                else:
                    invert=False
                    
                if len(split)>1:
                    use_cuts = np.array(split[1:]).astype(np.int32)
                else:
                    use_cuts=None
                
                self.apply_cuts(invert=invert, use_cuts=use_cuts)
                self.cutplot()

            elif x == 'r':
                if len(split)<2:
                    print("Select the index of the cut to remove")
                    ind = _cli_input(mode='integer')
                else:
                    ind = split[1]

                print(f"Removing cut {int(ind)}")
                self.current_subset.remove_cut(int(ind))
                self.apply_cuts()
                self.cutplot()
                                 
                    
            elif x == 'subset':
                if len(split)<2:
                    print("Select the index of the subset to switch to, or "
                          "enter 'new' to create a new subset.")
                    ind = _cli_input(mode='alpha-integer')
                else:
                    ind = split[1]
                    
                if ind == 'new':
                    ind = len(self.subsets)
                    print(f"Creating a new subset, index {ind}")
                    self.add_subset()
                
                print(f"Selecting subset {ind}")
                self.select_subset(int(ind))
                self.apply_cuts()
                self.cutplot()
                    
            else:
                print(f"Invalid input: {x}")
                
  
    def plot(self, axes=('X', 'Y'), log=False, clear=False, 
             xrange=None, yrange=None, zrange=None, 
             show=True, trim=True,
             figax = None):
        """
        Plots a histogram of the track data
        
        axes: tuple of str
            Passes to the axes parameter of 'frames'

        """
        
        if xrange is None:
            xrange = [None, None]
        if yrange is None:
            yrange = [None, None]
        if zrange is None:
            zrange = [None, None]
            
        fontsize=16
        ticksize=14
        
        xax, yax, arr = self.frames(axes=axes, trim=trim)

        # If a figure and axis are provided, use those
        if figax is not None:
            fig, ax = figax
        elif self.plotfig is None or clear:
            fig = plt.figure()
            ax = fig.add_subplot()
            self.plotfig = [fig, ax]
        else:
            fig, ax = self.plotfig
        
        if axes[0:2] == ('X', 'Y'):
            ax.set_aspect('equal')
            
        if len(axes) == 3:
            ztitle = axes[2]
            title = f"{axes[0]}, {axes[1]}, {axes[2]}"
        else:
            ztitle = '# Tracks'
            title = f"{axes[0]}, {axes[1]}"
            
            
        arr[arr==0] = np.nan
            
        # Calculate bounds
        if xrange[0] is None:
            xrange[0]  = np.nanmin(xax)
        if xrange[1] is None:
            xrange[1]  = np.nanmax(xax)
        if yrange[0] is None:
            yrange[0]  = np.nanmin(yax)
        if yrange[1] is None:
            yrange[1]  = np.nanmax(yax)

        if log:
            title += ' (log)'
            nonzero = np.nonzero(arr)
            arr[nonzero] = np.log10(arr[nonzero])

        else:
            title += ' (lin)'
            
        

        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        ax.set_xlabel(axes[0], fontsize=fontsize)
        ax.set_ylabel(axes[1], fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        
        try:
            p = ax.pcolorfast(xax, yax, arr.T)
            
            cb_kwargs = {'orientation':'vertical', 'pad':0.07, 'shrink':0.8, 'aspect':16}
            cbar= fig.colorbar(p, ax=ax, **cb_kwargs)
            cbar.set_label(ztitle, fontsize=fontsize)

        except ValueError: # raised if one of the arrays is empty
            pass
        
        
        
        
        
        if show:
            plt.show()
            
        return fig, ax
        
        
    def cutplot(self, clear=False):
        
        self.cutplotfig = plt.subplots(nrows=2, ncols=2, figsize=(9,9))
        self.cutplotfig[0].subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Figure tuple contains: 
        # (fig, axarr, bkg)
        fig, axarr = self.cutplotfig
        
        title = f"Subset {self.current_subset_i}, "
        if self.current_subset.current_dslice is None:
            title += 'All dslices selected'
        else:
            title += f"dslice {self.current_subset.current_dslice} selected"
        fig.suptitle(title)
                
        # X, Y
        ax = axarr[0][0]
        self.plot(axes=('X', 'Y'), show=False, trim=True, figax = (fig, ax),
                  xrange=self.current_subset.domain.xrange,
                  yrange=self.current_subset.domain.yrange)
        
        # D, C
        ax = axarr[0][1]
        self.plot(axes=('D', 'C'), show=False, figax = (fig, ax),
                   log=True, trim=False,
                   xrange=self.current_subset.domain.drange,
                   yrange=self.current_subset.domain.crange)
        
        # X, Y, D
        ax = axarr[1][0]
        self.plot(axes=('X', 'Y', 'D'), show=False, trim=False, figax = (fig, ax),
                   xrange=self.current_subset.domain.xrange,
                   yrange=self.current_subset.domain.yrange,
                   zrange=self.current_subset.domain.drange)
        
        
        # D, E
        ax = axarr[1][1]
        self.plot(axes=('D', 'E'),  show=False, trim=True, figax = (fig, ax),
                   log=True,
                   xrange=self.current_subset.domain.drange,
                   yrange=self.current_subset.domain.erange)
        
        plt.show()
        
        return fig, ax
        
  
        
  


if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('//expdiv','kodi','ShotData')
    #data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    obj = CR39(103955, data_dir=data_dir, verbose=True)
    
    #domain = Cut(xmin=-5, xmax=0)
    #subset = Subset(domain=domain)
    
    #obj.add_subset(subset)
    #obj.add_cut(Cut(cmin=40))
    

    #obj.cli()
    
    """
    path = os.path.join(os.getcwd(), 'testcr39.h5')
    print(path)
    obj.save(path)
    
    c2 = CR39(103955, data_dir=data_dir, subsets=path)
    print(c2.subsets)
    """
        
        
        
        
            
