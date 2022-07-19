# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Object representing a CR39 dataset
"""

import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt


FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])

class CR39:
    
    def __init__(self, path, verbose=False):
        self.verbose = verbose
        
        self.cuts = []
        
        
        self._read_CPSA(path)
        
    def _log(self, msg):
        if self.verbose:
            print(msg)
        
        
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
                #
                # The x and y pos are relative to the upper right corner 
                # of the current frame
                
                t= np.zeros([fh.hits, 6])
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
    
                    
                tracks.append(t)
        self._log("Done Reading CPSA data")
    
        
        self._log("Processing the tracks")
        


    
        # Sort the yaxis (it's backwards...)
        self.yax = np.sort(self.yax)

        
        # Re-shape the track data into a list of every track
        self.trackdata = np.zeros([tot_hits, 6])
        for i in range(self.nframes):
            self.trackdata[cum_hits[i]:cum_hits[i+1], :] = tracks[i]

            
    def frames(self):

        arr, xedges, yedges = np.histogram2d(self.trackdata[:,0],
                                             self.trackdata[:,1],
                                             bins=[self.xax, self.yax])
        # Calculate the bin centers
        xax =0.5*(xedges[:-1] + xedges[1:])
        yax =0.5*(yedges[:-1] + yedges[1:])
        
        return xax, yax, arr
        
        
        
        
    def plot(self):
        
        xax, yax, arr = self.frames()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(xax, yax, arr.T, vmax=25*np.median(arr))
        
        
        
        
        
            
