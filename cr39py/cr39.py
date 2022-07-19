# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Object representing a CR39 dataset
"""

import numpy as np

from collections import namedtuple


FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])

class CR39:
    
    def __init__(self, path, verbose=False):
        
        self._read_CPSA(path)
        
        self.verbose = verbose
        
        
    def _log(self, msg):
        if self.verbose:
            self.print(msg)
        
        
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
            frame_headers = []
            # Tracks in each frame
            tracks = []
            # Keep track of the number of hits in each frame, and total
            cum_hits = np.array([0], dtype='int32')
            tot_hits = 0
            
            
            for i in range(self.nframes):
                if i % 5000 == 4999:
                    self._log(f"Reading frame {i+1}/{self.nframes}")
                
                # Read the bin header
                h = np.fromfile(file, count=10, dtype='int32' )

                fh = FrameHeader(number=h[0], xpos=h[1], ypos=h[2], hits=h[3],
                                 BLENR=h[4:7], zpos=h[7], x_ind=h[8],
                                 y_ind = h[9])
                frame_headers.append(fh)
                
                # Increment the counters for the number of hits
                tot_hits += fh.hits
                cum_hits = np.append(cum_hits, int(fh.hits))
                
                # Read the track data for this frame
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
                    t[:, 0] = np.fromfile(file, count=fh.hits, 
                                          dtype='int16')*self.pix_size*1e-4
                    # Adjustment for ?
                    t[:, 0] += self.fx*0.5e-4 + fh.xpos*1e-5
                    
                     # y position, cm
                    t[:, 1] = np.fromfile(file, count=fh.hits, 
                                          dtype='int16')*self.pix_size*1e-4
                    # Adjustment for ?
                    t[:, 1] += -self.fy*0.5e-4 + fh.ypos*1e-5
                    
                tracks.append(t)
        self._log("Done Reading CPSA data")
        
        self._log("Processing the tracks")
        # Re-shape the track data into a list of every track
        self.trackdata = np.zeros([tot_hits, 6])
        for i in range(self.nframes):
            self.trackdata[cum_hits[i]:cum_hits[i]+cum_hits[i+1]] = tracks[i]
            
