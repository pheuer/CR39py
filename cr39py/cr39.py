# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Object representing a CR39 dataset
"""
import os
import numpy as np

from collections import namedtuple

from fast_histogram import histogram2d
import matplotlib.pyplot as plt

from cr39py.util.misc import find_file


# Represents metadata from a single frame of cr39
# used when reading CPSA files
FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])



class Cut:
    defaults = {'xmin':-1e6, 'xmax':1e6, 'ymin':-1e6, 'ymax':1e6,
                'dmin':0, 'dmax':1e6, 'cmin':0, 'cmax':1e6, 
                'emin':0, 'emax':1e6}
    
    indices = {'xmin':0, 'xmax':0, 'ymin':1, 'ymax':1,
                'dmin':2, 'dmax':2, 'cmin':3, 'cmax':3, 
                'emin':5, 'emax':5}

    
    def __init__(self,  xmin : float = None, xmax : float = None,
                        ymin : float = None, ymax : float = None,
                        dmin : float = None, dmax : float = None,
                        cmin : float = None, cmax : float = None,
                        emin : float = None, emax : float = None):
        
        self.dict = {'xmin':xmin, 'xmax':xmax,
                     'ymin':ymin, 'ymax':ymax,
                     'dmin':dmin, 'dmax':dmax,
                     'cmin':cmin, 'cmax':cmax,
                     'emin':emin, 'emax':emax}

    def __getattr__(self, key):
        
        if key in self.dict.keys():
            if self.dict[key] is None:
                return self.defaults[key]
            else:
                return self.dict[key]
        else:
            raise ValueError(f"Unknown attribute for Cut: {key}")
            
            
    @property
    def xrange(self):
        return [self.dict['xmin'], self.dict['xmax']]
    @property
    def yrange(self):
        return [self.dict['ymin'], self.dict['ymax']]
    @property
    def drange(self):
        return [self.dict['dmin'], self.dict['dmax']]
    @property
    def crange(self):
        return [self.dict['cmin'], self.dict['cmax']]
    @property
    def erange(self):
        return [self.dict['emin'], self.dict['emax']]
                
            
            
    def empty(self):
        return all(v is None for k,v in self.dict.items())
            
            
            
    def __str__(self):
        s = [f"{key}:{val}" for key, val in self.dict.items() if val is not None ]
        s = ', '.join(s)
        return s
    
    
    def test (self, trackdata):
        """
        Given tracks, return a boolean array representing which tracks
        fall within this cut
        """
        ntracks, _ = trackdata.shape
        keep = np.ones(ntracks).astype('bool')

        for key in self.dict.keys():
            if self.dict[key] is not None:
                i = self.indices[key]
                if 'min' in key:
                    keep *= np.greater(trackdata[:, i], getattr(self, key))
                else:
                    keep *= np.less(trackdata[:, i], getattr(self, key))      
        
        # Return a 1 for every track that is in the cut
        return keep.astype(bool)
        
        
        
        
        
def _cli_input(mode='alphanumeric list'):
    """
    Collects CLI input from the user: continues asking until a valid
    input has been provided. Input modes: 
        
    'numeric'
        Single number
        
    'alphanumeric list'
        List of alphanumeric characters, separated by commas    
    
    'key:value list' -> e.g. xmin:10, ymin:-1
        Alternating alpha and numeric key value pairs, comma separated
        Letters are only acceptable in the values if they are 'none'
    """
    integers = set('123456790+-')
    floats = integers.union(".e")
    alphas = set('abcdefghijklmnopqrstuvwxyz')

    while True:
        x = str(input(">"))
        
        if mode=='integer':
            if x in integers:
                return x

        elif mode=='alpha-integer list':
            split = x.split(',')
            split = [s.strip() for s in split]
            # Discard empty strings
            split = [s for s in split if s!='']
            if all([set(s).issubset( alphas.union(integers)) for s in split ]):
                return split
                              
        elif mode == 'key:value list':
            split = x.split(',')
            split = [ s.split(':') for s in split]
            
            # Verify that we have a list of at least one pair, and only pairs
            if all([len(s)==2 for s in split]) and len(split)>0:
                # Discard empty strings
                split = [s for s in split if (s[0]!='' and s[1]!='')]
                
                # Transform any 'none' values into None
                # Strip any other values
                for i, s in enumerate(split):
                    if str(s[1]).lower() == 'none':
                        split[i][1] = None
                    else:
                        split[i][1] = s[1].strip()
                
                # Test that values are in the correct sets
                test1 = all([(
                            (set(s[0].strip()).issubset(alphas)) and
                             (s[1] is None or set(s[1]).issubset(floats))
                             )
                             for s in split])
                
                # Convert any non-None values into floats
                for i, s in enumerate(split):
                    if s[1] is not None:
                        split[i][1] = float(s[1])
                
                if all([test1,]):
                    return {str(s[0].strip()):s[1] for s in split}
        else:
            raise ValueError("Invalid Mode")
        
                        

class CR39:

    # Axes dictionary for trackdata
    axes_ind = {'X':0, 'Y':1, 'D':2, 'C':3, 'E':5, 'Z':6}
    ind_axes = ['X', 'Y', 'D', 'C', 'E', 'Z']
    
    def __init__(self, *args, verbose=False, data_dir=None, domain=None):
        
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
            
        domain : Cut
            A Cut that is interpreted as inclusive, e.g. sets the domain for 
            the remainder of the 
            
        """
        self.verbose = verbose
        
        self.cuts = []
        if domain is not None:
            self.set_domain(domain)
        else:
            self.domain=None

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
        
    def _log(self, msg):
        if self.verbose:
            print(msg)
            
            
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
            
        self.ntracks = tot_hits
        # Store all the tracks, as a starting point for making cuts on
        # self.trackdata
        self.raw_trackdata = np.copy(self.trackdata)
        
        # Sort the yaxis (it's backwards...)
        self.yax = np.sort(self.yax)
        
        # Make axes for the other quantites
        self.dax = np.linspace(0, 20, num=40)
        self.cax = np.linspace(0, 80, num=80)
        self.eax = np.linspace(0, 50, num=50)
        
        

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
        
        
        
    def set_domain(self, cut):
        """
        Sets the domain cut: an inclusive cut that will not be inverted
        """
        self.domain = cut
        
    
    
    def add_cut(self, c):
        self.cuts.append(c)
           
    def remove_cut(self, i):
        if i > len(self.cuts)-1:
            raise ValueError(f"Cannot remove the {i} cut, there are only "
                             f"{len(self.cuts)} cuts.")
        self.cuts.pop(i)
        
    def replace_cut(self, i, c):
        self.cuts[i] = c
        
    def print_cuts(self):
        if len(self.cuts) == 0:
            print("No cuts")
        
        for i,cut in enumerate(self.cuts):
            print(f"Cut {i}: {str(cut)}")
        
        
    def apply_cuts(self, subset=None, invert=False):
        """
        Apply cuts to the track data
        
        subset : int, list of ints (optional)
            If provided, only the cuts corresponding to the int or ints 
            provided will be applied. The default is to apply all cuts
            
        invert : bool (optional)
            If true, return the inverse of the cuts selected. Default is 
            false.
            

        """

        valid_cuts = list(np.arange(len(self.cuts)))
        if subset is None:
            subset = valid_cuts
        else:
            for s in subset:
                if s not in valid_cuts:
                    raise ValueError(f"Specified cut index is invalid: {s}")
        subset = list(subset)    

        keep = np.ones(self.ntracks).astype(bool)   
        
        for i, cut in enumerate(self.cuts):
            if i in subset:
                print(f"Applying cut {i}")
                
                # Get a boolean array of tracks that are inside this cut
                x = cut.test(self.raw_trackdata)
                
                # negate to get a list of tracks that are NOT
                # in the excluded region (unless we are inverting)
                if not invert:
                    x = np.logical_not(x)
                    
                keep *= x
                
        # Regardless of anything else, only show tracks that are within
        # the domain
        if self.domain is not None:
            keep *= self.domain.test(self.raw_trackdata)
                
        self.trackdata = self.raw_trackdata[keep, :]
            
            
    
    def cutcli(self):
        print("enter 'help' for a list of commands")
        self.apply_cuts()
        self.cutplot()
        
        while True:
            
            print("Domain:")
            if self.domain is None:
                print("No domain set: default is the full dataset")
            else:
                print(str(self.domain))
            
            print("Current cuts:")
            if len(self.cuts) == 0:
                print("No cuts set yet")
            else:
                for i, cut in enumerate(self.cuts):
                    print(f"{i} : " + str(cut))
            
            print("add (a), edit (e), edit the domain (d), remove (r), plot (p), "
                  "plot inverse (pi), help (help)")
            split = _cli_input(mode='alpha-integer list')
            x = split[0]
            
            if x == 'help':
                print("Enter commands, followed by any additional arugments "
                      "separated by commas.\n"
                      " ** Commands ** \n"
                      "'a' -> create a new cut\n"
                      "'d' -> edit the domain\n"
                      "'e' -> edit a cut\n"
                      "Argument (one int) is the cut to edit\n"
                      "'r' -> remove an existing cut\n"
                      "Arguments are numbers of cuts to remove\n"
                      "'p' -> plot the image with current cuts\n"
                      "Arguments are numbers of cuts to include in plot\n"
                      "The default is to include all of the current cuts\n"
                      "'pi' -> plot the image with INVERSE of the cuts\n"
                      "Arguments are numbers of cuts to include in plot\n"
                      "The default is to include all of the current cuts\n"
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
                        self.replace_cut(ind, c)
                    else:
                        self.add_cut(c)
                        
                self.apply_cuts()
                self.cutplot()
                
            elif x == 'd':
                print("Current domain: " + str(self.domain))
                print("Enter a list key:value pairs with which to modify the domain"
                      "(set a key to 'None' to remove it)")
                kwargs = _cli_input(mode='key:value list')
                for key in kwargs.keys():
                    if str(kwargs[key]).lower() == 'none':
                        self.domain.dict[key] = None
                    else:
                        self.domain.dict[key] = float(kwargs[key])
                self.apply_cuts()
                self.cutplot()
 
                        
            elif x == 'e':
                if len(split)>1:
                    ind = int(split[1])
                    cut = self.cuts[ind]
                    
                    print(f"Selected cut ({ind}) : " + str(self.cuts[ind]))
                    print("Enter a list key:value pairs with which to modify this cut"
                          "(set a key to 'None' to remove it)")
                    
                    kwargs = _cli_input(mode='key:value list')
                    for key in kwargs.keys():
                        if str(kwargs[key]).lower() == 'none':
                            self.cuts[ind].dict[key] = None
                        else:
                            self.cuts[ind].dict[key] = float(kwargs[key])
                            
                    self.apply_cuts()
                    self.cutplot()
                else:
                    print("Specify the number of the cut you want to modify "
                          "as an argument after the command.")
                        
                        
            elif x in ['p', 'pi']:
                if x =='pi':
                    invert=True
                else:
                    invert=False
                    
                if len(split)>1:
                    subset = np.array(split[1:]).astype(np.int32)
                else:
                    subset=None
                
                self.apply_cuts(invert=invert, subset=subset)
                self.cutplot()

            elif x == 'r':
                if len(split)>1:
                    for i in split[1:]:
                        print(f"Removing cut {int(i)}")
                        self.remove_cut(int(i))
                else:
                    print("Specify which cuts to remove as arguments after the command.")
                

            else:
                print(f"Invalid input: {x}")
                
  
    def plot(self, axes=('X', 'Y'), log=False, clear=False, 
             xrange=[None, None], yrange=[None, None], zrange=[None, None], 
             show=True, trim=True,
             figax = None):
        """
        Plots a histogram of the track data
        
        axes: tuple of str
            Passes to the axes parameter of 'frames'

        """
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
        
        p = ax.pcolorfast(xax, yax, arr.T)
        
        cb_kwargs = {'orientation':'vertical', 'pad':0.07, 'shrink':0.8, 'aspect':16}
        cbar= fig.colorbar(p, ax=ax, **cb_kwargs)
        cbar.set_label(ztitle, fontsize=fontsize)
        
        
        
        
        if show:
            plt.show()
            
        return fig, ax
        
        
    def cutplot(self, clear=False):
        

        self.cutplotfig = plt.subplots(nrows=2, ncols=2, figsize=(9,9))
        self.cutplotfig[0].subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Figure tuple contains: 
        # (fig, axarr, bkg)
        fig, axarr = self.cutplotfig
                
        # X, Y
        ax = axarr[0][0]
        self.plot(axes=('X', 'Y'), show=False, trim=True, figax = (fig, ax),
                  xrange=self.domain.xrange,
                  yrange=self.domain.yrange)
        
        # D, C
        ax = axarr[0][1]
        self.plot(axes=('D', 'C'), show=False, figax = (fig, ax),
                   log=True, trim=False,
                   xrange=self.domain.drange,
                   yrange=self.domain.crange)
        
        # X, Y, D
        ax = axarr[1][0]
        self.plot(axes=('X', 'Y', 'D'), show=False, trim=False, figax = (fig, ax),
                   xrange=self.domain.xrange,
                   yrange=self.domain.yrange,
                   zrange=self.domain.drange)
        
        
        # D, E
        ax = axarr[1][1]
        self.plot(axes=('D', 'E'),  show=False, trim=True, figax = (fig, ax),
                   log=True,
                   xrange=self.domain.drange,
                   yrange=self.domain.erange)
        
        plt.show()
        
        return fig, ax
        
  
        
  
"""
import os
from cr39py.cr39 import *
path = os.path.join('\\\expdiv','kodi','ShotData','104394', '104394_TIM2_PR2709_2h_s4.cpsa')
obj = CR39(path, verbose=True)
"""


if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('//expdiv','kodi','ShotData')
    #data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    
    domain = Cut(xmin=-5, xmax=0)
    obj = CR39(103955, data_dir=data_dir, verbose=True, domain=domain)
    
    #obj.add_cut(Cut(cmin=40))
    

    obj.cutcli()
        
        
        
        
            
