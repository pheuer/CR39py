# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Adapted in part from code written by Hans Rinderknecht
"""
from pathlib import Path
import h5py
import numpy as np
import os

from collections import namedtuple

from fast_histogram import histogram2d
import matplotlib.pyplot as plt

from cr39py.util.cli import _cli_input
from cr39py.cut import Cut
from cr39py.subset import Subset

from cr39py.response import track_energy

from cr39py.util.units import unit_registry as u

from cr39py.util.baseobject import BaseObject


__all__ = [
    "Scan",
]


# Represents metadata from a single frame of cr39
# used when reading CPSA files
FrameHeader = namedtuple(
    "FrameHeader", ["number", "xpos", "ypos", "hits", "BLENR", "zpos", "x_ind", "y_ind"]
)


class Scan(BaseObject):

    # Axes dictionary for trackdata
    axes_ind = {"X": 0, "Y": 1, "D": 2, "C": 3, "E": 5, "Z": 6}

    ax_units = {
        "X": u.cm,
        "Y": u.cm,
        "D": u.um,
        "C": u.dimensionless,
        "E": u.dimensionless,
        "Z": u.um,
    }

    def __init__(self, verbose=False):
        super().__init__()

        _exportable_attributes = ["current_subset_index", "subsets", "axes"]
        self._exportable_attributes += _exportable_attributes

        self.verbose = True

        # Store figures once created for blitting
        self.plotfig = None
        self.cutplotfig = None

        self.current_subset_index = 0
        self.subsets = []
        self.axes = {"X": 0, "Y": 0, "D": 0, "C": 0, "E": 0, "Z": 0}

        self.trackdata = None
        self.trackdata_subset = None

    def _write_hdf5_contents(self, path, group=None):
        super()._write_hdf5_contents(path, group=group)

        if group is None:
            group = "/"

        # Save the track data
        with h5py.File(path, "a") as f:
            grp = f[group]
            trackgrp = grp.require_group("tracks")
            trackgrp.create_dataset(
                "trackdata",
                self.trackdata.shape,
                compression="gzip",
                compression_opts=3,
                dtype="f4",
            )

            trackgrp["trackdata"][...] = self.trackdata.astype(np.dtype("f4"))

    def _read_hdf5_contents(self, path, group=None):
        super()._read_hdf5_contents(path, group=group)

        if group is None:
            group = "/"

        # Load the track data
        with h5py.File(path, "r") as f:
            grp = f[group]["tracks"]
            self.trackdata = grp["trackdata"][...]

        self.apply_cuts()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @classmethod
    def from_cpsa(cls, path, verbose=None):
        obj = cls(verbose=verbose)
        obj._read_cpsa(path)
        return obj

    def _read_cpsa(self, path):

        print("Reading .cpsa file")

        with open(path, "rb") as file:

            # First read in the header values
            self._log("Reading CPSA header")
            self.version = -np.fromfile(file, count=1, dtype="int32")[0]
            self._log(f"Version: {self.version}")

            # Number of microscope frames in the x and y directions of the scan
            # respectively
            self.nx = np.fromfile(file, count=1, dtype="int32")[0]
            self.ny = np.fromfile(file, count=1, dtype="int32")[0]
            self.nframes = self.nx * self.ny
            self._log(f"nx, ny microscope bins: {self.nx}, {self.ny}")
            self._log(f"Nframes: {self.nframes}")

            # h[3] is "Nbins" which is not used except with legacy data\
            Nbins = np.fromfile(file, count=1, dtype="int32")[0]

            # Pixel size in microns: note that this is stored as a single
            # so needs to be read as such
            self.pix_size = np.fromfile(file, count=1, dtype="single")[0]
            self._log(f"Pixel size: {self.pix_size:.1e} um")

            # h[5] is "ppb" which is not used except with legacy data
            pbb = np.fromfile(file, count=1, dtype="single")[0]

            # Thresholds for border, contrast, eccentricity,
            # and number of eccentricity moments
            self.threshold = np.fromfile(file, count=4, dtype="int32")[0]
            self._log(f"Threshold: {self.threshold}")

            # Number of utilized camera image pixels in the x and y directions
            self.NFPx = np.fromfile(file, count=1, dtype="int32")[0]
            self.NFPy = np.fromfile(file, count=1, dtype="int32")[0]
            self._log(f"Untilized camera image px NFPx, NFPy: {self.NFPx}, {self.NFPy}")

            # Microscope frame size in microns
            self.fx = self.pix_size * self.NFPx
            self.fy = self.pix_size * self.NFPy
            self._log(
                f"Microscope frame size fx, fy: {self.fx:.1e} um, {self.fy:.1e} um"
            )

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
            cum_hits = np.array([0], dtype="int32")

            # Collect x and y positions of frames in sets that, once sorted
            # will be the x and y axes of the dataset
            self.axes["X"] = np.zeros(self.nx)
            self.axes["Y"] = np.zeros(self.ny)
            for i in range(self.nframes):
                if i % 5000 == 4999:
                    self._log(f"Reading frame {i+1}/{self.nframes}")

                # Read the bin header
                h = np.fromfile(file, count=10, dtype="int32")

                # Header contents are as follows
                # 0 -> frame number (starts at 1)
                # 1 -> xpos (frame center x position, in 1e-7 m)
                # 2 -> ypos (frame center y position, in 1e-7 m)
                # 3 -> hits (number of tracks in this frame)
                # 4,5,6 -> BLENR (something about rejected tracks?)
                # 7 -> zpos (microscope focus? Units? )
                # 8 -> x_ind (x index of the frame, staring at 0)
                # 9 -> y_ind (y index of the frame, staring at 0)

                fh = FrameHeader(
                    number=h[0],
                    xpos=h[1],
                    ypos=h[2],
                    hits=h[3],
                    BLENR=h[4:7],
                    zpos=h[7],
                    x_ind=h[8],
                    y_ind=h[9],
                )
                self.frame_headers.append(fh)

                # Put the bin x and y values in the appropriate place in the axes
                self.axes["X"][fh.x_ind] = fh.xpos * 1e-5
                self.axes["Y"][fh.y_ind] = fh.ypos * 1e-5

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

                t = np.zeros([fh.hits, 7])
                if fh.hits > 0:

                    # Diameters (converting to um)
                    t[:, 2] = (
                        np.fromfile(file, count=fh.hits, dtype="int16")
                        * 1e-2
                        * self.pix_size
                    )

                    # Ecentricities
                    t[:, 5] = np.fromfile(file, count=fh.hits, dtype="byte")

                    # Contrast
                    t[:, 3] = np.fromfile(file, count=fh.hits, dtype="byte")

                    # Avg Contrast
                    t[:, 4] = np.fromfile(file, count=fh.hits, dtype="byte")

                    # x position, cm
                    # Positions are relative to the top right of the current
                    # frame, so we need to adjust them accordingly
                    t[:, 0] = (
                        -np.fromfile(file, count=fh.hits, dtype="int16")
                        * self.pix_size
                        * 1e-4
                        + fh.xpos * 1e-5
                        + (self.fx / 2) * 1e-4
                    )

                    # y position, cm
                    t[:, 1] = (
                        np.fromfile(file, count=fh.hits, dtype="int16")
                        * self.pix_size
                        * 1e-4
                        + fh.ypos * 1e-5
                        - (self.fy / 2) * 1e-4
                    )

                    # z position, microns
                    t[:, 6] = fh.zpos * self.pix_size * 1e-2

                tracks.append(t)
        self._log("Done Reading CPSA data")

        self._log("Processing the tracks")

        # The order of the quantities in track data is:
        # 0) x position (cm))
        # 1) y position (cm)
        # 2) diameter (um)
        # 3) contrast (dimless)
        # 4) avg contrast (dimless)
        # 5) ecentricity (dimless)
        # 6) z position/lens position (um)

        # Re-shape the track data into a list of every track
        self.trackdata_subset = np.zeros([tot_hits, 7])
        for i in range(self.nframes):
            self.trackdata_subset[cum_hits[i] : cum_hits[i + 1], :] = tracks[i]

        # Sort the tracks by diameter for later slicing into energy dslices
        isort = np.argsort(self.trackdata_subset[:, 2])
        self.trackdata_subset = self.trackdata_subset[isort, :]

        # Store all the tracks, as a starting point for making cuts on
        # self.trackdata_subset
        self.trackdata = np.copy(self.trackdata_subset)

        # Sort the yaxis (it's backwards...)
        self.axes["Y"] = np.sort(self.axes["Y"])

        # Make axes for the other quantites
        self.axes["D"] = np.linspace(0, 20, num=40)
        self.axes["C"] = np.linspace(0, 80, num=80)
        self.axes["E"] = np.linspace(0, 50, num=60)

        # Initialize the list of subsets with a single subset to start.
        self.subsets = [
            Subset(),
        ]

    def set_binwidth(self, axis, binwidth):
        """
        Sets the bin width for a given axis
        """
        axis = axis.upper()
        if axis in self.axes.keys():
            amin, amax = np.min(self.axes[axis]), np.max(self.axes[axis])
            self.axes[axis] = np.arange(amin, amax, binwidth)

    @property
    def ntracks(self):
        return self.trackdata_subset.shape[0]

    @property
    def raw_ntracks(self):
        return self.trackdata.shape[0]

    def frames(self, axes=("X", "Y"), trim=True, hax=None, vax=None):
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
            'Z' : z position/lens position during scan

            The default is ('X', 'Y')

        trim : bool
            If true, trim the array and axes down to exclude any entirely
            empty rows or columns


        hax, vax : np.ndarrays
            If set, replaces the default axes

        """
        i0 = self.axes_ind[axes[0]]
        i1 = self.axes_ind[axes[1]]

        if hax is None:
            ax0 = self.axes[axes[0]]
        else:
            ax0 = hax

        if vax is None:
            ax1 = self.axes[axes[1]]
        else:
            ax1 = vax

        # If creating a histogram like the X,Y,D plots
        if len(axes) == 3:
            i2 = self.axes_ind[axes[2]]
            weights = self.trackdata_subset[:, i2]
        else:
            weights = None

        # If the implied range is < 25x the spacing of the axis,
        # create a higher resolution axis that spans this range
        dax0 = np.mean(np.gradient(ax0))
        if (np.max(ax0) - np.min(ax0)) < 10 * dax0:
            ax0 = np.linspace(np.min(ax0), np.max(ax0), num=30)
        dax1 = np.mean(np.gradient(ax1))
        if (np.max(ax1) - np.min(ax1)) < 10 * dax1:
            ax1 = np.linspace(np.min(ax1), np.max(ax1), num=30)

        rng = [(np.min(ax0), np.max(ax0)), (np.min(ax1), np.max(ax1))]
        bins = [ax0.size, ax1.size]

        xax = np.linspace(rng[0][0], rng[0][1], num=bins[0])
        yax = np.linspace(rng[1][0], rng[1][1], num=bins[1])

        arr = histogram2d(
            self.trackdata_subset[:, i0],
            self.trackdata_subset[:, i1],
            bins=bins,
            range=rng,
            weights=weights,
        )

        # Create the unweighted histogram and divide by it (sans zeros)
        if len(axes) == 3:
            arr_uw = histogram2d(
                self.trackdata_subset[:, i0],
                self.trackdata_subset[:, i1],
                bins=bins,
                range=rng,
            )
            nz = np.nonzero(arr_uw)
            arr[nz] = arr[nz] / arr_uw[nz]

        # Get rid of any entirely zero rows or columns
        if trim:
            xi = np.nonzero(np.sum(arr, axis=1))[0]
            if len(xi) == 0:
                xa, xb = 0, -1
            else:
                xa, xb = xi[0], xi[-1] + 1

            yi = np.nonzero(np.sum(arr, axis=0))[0]
            if len(yi) == 0:
                ya, yb = 0, -1
            else:
                ya, yb = yi[0], yi[-1] + 1

            xax = xax[xa:xb]
            yax = yax[ya:yb]
            arr = arr[xa:xb, ya:yb]

        return xax, yax, arr

    def hreflect(self):
        self.trackdata[:, 0] *= -1
        self.trackdata_subset[:, 0] *= -1
        self.plot()

    def vreflect(self):
        self.trackdata[:, 1] *= -1
        self.trackdata_subset[:, 1] *= -1
        self.plot()

    # ************************************************************************
    # Methods for managing subset list
    # ************************************************************************

    @property
    def current_subset(self):
        return self.subsets[self.current_subset_index]

    @property
    def nsubsets(self):
        return len(self.subsets)

    def select_subset(self, i):
        if i > self.nsubsets - 1 or i < -self.nsubsets:
            raise ValueError(
                f"Cannot select subset {i}, there are only " f"{self.nsubsets} subsets."
            )
        else:
            # Handle negative indexing
            if i < 0:
                i = self.nsubsets + i
            self.current_subset_index = i

    def add_subset(self, subset):
        self.subsets.append(subset)

    def remove_subset(self, i):
        if i > self.nsubsets - 1:
            raise ValueError(
                f"Cannot remove the {i} subset, there are only "
                f"{self.subsets} subsets."
            )
        else:
            self.subsets.pop(i)

    # ***********************************************************
    # Cut functions
    # These all apply to the current subset and are just wrappers
    # for the matching method on that class
    # ***********************************************************
    def add_cut(self, c):
        self.current_subset.add_cut(c)
        self.apply_cuts()

    def remove_cut(self, i):
        self.current_subset.remove_cut(i)
        self.apply_cuts()

    def replace_cut(self, i, c):
        self.current_subset.replace_cut(i, c)
        self.apply_cuts()

    def apply_cuts(self, use_cuts=None, invert=False):
        """
        Apply currently selected cuts and dslices to the track data

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
                x = cut.test(self.trackdata)

                # negate to get a list of tracks that are NOT
                # in the excluded region (unless we are inverting)
                if not invert:
                    x = np.logical_not(x)
                keep *= x

        # Regardless of anything else, only show tracks that are within
        # the domain
        if self.current_subset.domain is not None:
            keep *= self.current_subset.domain.test(self.trackdata)

        # Select only these tracks
        self.trackdata_subset = self.trackdata[keep, :]

        # Calculate the bin edges for each dslice
        # !! note that the tracks are already sorted into order by diameter
        # when the CR39 data is read in
        if self.current_subset.ndslices != 1:
            # Figure out the dslice width
            dbin = int(self.trackdata_subset.shape[0] / self.current_subset.ndslices)
            # Extract the appropriate portion of the tracks
            b0 = self.current_subset.current_dslice_index * dbin
            b1 = b0 + dbin
            self.trackdata_subset = self.trackdata_subset[b0:b1, :]

    def save_tracks(self, path, group=None):
        """
        Save the list of tracks (with cuts applied) to an h5 file
        """
        if group is None:
            group = "/"

        self.apply_cuts()

        with h5py.File(path, "a") as f:
            grp = f[group]

            if "trackdata" in grp.keys():
                del grp["trackdata"]

            grp.create_dataset(
                "trackdata",
                data=self.trackdata_subset,
                compression="gzip",
                compression_opts=5,
            )

            # Attach a string describing what each of the values in the
            # list represents
            desc = (
                "x(um), y(um), diameter(cm), contrast(dimless), "
                "avg contrast(dimless),ecentricity(dimless), "
                "scanning z position (um)"
            )
            grp["trackdata"].attrs["description"] = desc

        # Save the current subset info
        self.current_subset._save(path, group=group + "/cuts")

    def avg_energy(self, etch_time, particle):
        """
        The average energy of the tracks on the current slice
        """

        avg_d = np.mean(self.trackdata_subset[:, 2])

        avg_e = track_energy(avg_d, particle, etch_time)

        return avg_e

    def cli(self):
        self.apply_cuts()
        self.cutplot()

        # This flag keeps track of whether any changes have been made
        # by the CLI, and will be returned when it exits
        changed = False

        while True:

            print("*********************************************************")
            print(
                f"Current subset index: {self.current_subset_index} of {np.arange(len(self.subsets))}"
            )
            # Print a summary of the current subset
            print(self.current_subset)
            print(
                f"ntracks selected: {self.ntracks:.1e} " f"(of {self.raw_ntracks:.1e})"
            )

            print(
                "add (a), edit (e), edit the domain (d), remove (r), plot (p), "
                "plot inverse (pi), switch subsets (subset), change dslices (dslice), "
                "change the number of dslices (ndslices), end (end), help (help)"
            )

            split = _cli_input(mode="alpha-integer list", always_pass=[])
            x = split[0]

            if x == "help":
                print(
                    "Enter commands, followed by any additional arugments "
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
                    "'binwidth` -> Change the binwidth on an axis\n"
                    " ** Cut keywords ** \n"
                    "xmin, xmax, ymin, ymax, dmin, dmax, cmin, cmax, emin, emax\n"
                    "e.g. 'xmin:0,xmax:5,dmax=15'\n"
                )

            elif x == "end":
                self.apply_cuts()
                self.cutplot()
                break

            elif x == "a":
                print("Enter new cut parameters as key:value pairs separated by commas")
                kwargs = _cli_input(mode="key:value list")

                # validate the keys are all valid dictionary keys
                valid = True
                for key in kwargs.keys():
                    if key not in list(Cut.defaults.keys()):
                        print(f"Unrecognized key: {key}")
                        valid = False

                if valid:
                    c = Cut(**kwargs)
                    if x == "r":
                        ind = int(split[1])
                        self.current_subset.replace_cut(ind, c)
                    else:
                        self.current_subset.add_cut(c)

                self.apply_cuts()
                self.cutplot()
                changed = True

            elif x == "binwidth":
                print("Enter the name of the axis to change")
                ax_name = _cli_input(mode="alpha-integer")
                ax_name = ax_name.upper()
                print(f"Selected axis {ax_name}")
                binwidth = np.mean(np.gradient(self.axes[ax_name]))
                print(f"Current binwidth is {binwidth:.1e}")
                print("Enter new binwidth")
                binwidth = _cli_input(mode="float")
                self.set_binwidth(ax_name, binwidth)

                self.apply_cuts()
                self.cutplot()
                changed = True

            elif x == "dslice":
                if len(split) < 2:
                    print(
                        "Select the index of the dslice to switch to, or"
                        "enter 'all' to select all dslices"
                    )
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]

                if ind == "all":
                    self.current_subset.select_dslice(None)
                else:
                    self.current_subset.select_dslice(int(ind))
                self.apply_cuts()
                self.cutplot()
                changed = True

            elif x == "d":
                print("Current domain: " + str(self.current_subset.domain))
                print(
                    "Enter a list key:value pairs with which to modify the domain"
                    "(set a key to 'None' to remove it)"
                )
                kwargs = _cli_input(mode="key:value list")
                for key in kwargs.keys():
                    if str(kwargs[key]).lower() == "none":
                        self.current_subset.domain.bounds[key] = None
                    else:
                        self.current_subset.domain.bounds[key] = float(kwargs[key])
                self.apply_cuts()
                self.cutplot()
                changed = True

            elif x == "e":
                if len(split) > 1:
                    ind = int(split[1])

                    print(
                        f"Selected cut ({ind}) : " + str(self.current_subset.cuts[ind])
                    )
                    print(
                        "Enter a list key:value pairs with which to modify this cut"
                        "(set a key to 'None' to remove it)"
                    )

                    kwargs = _cli_input(mode="key:value list")
                    for key in kwargs.keys():
                        if str(kwargs[key]).lower() == "none":
                            self.current_subset.cuts[ind].bounds[key] = None
                        else:
                            self.current_subset.cuts[ind].bounds[key] = float(
                                kwargs[key]
                            )

                    self.apply_cuts()
                    self.cutplot()
                    changed = True
                else:
                    print(
                        "Specify the number of the cut you want to modify "
                        "as an argument after the command."
                    )

            elif x == "ndslices":
                if len(split) < 2:
                    print("Enter the requested number of dslices")
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]
                self.current_subset.set_ndslices(int(ind))
                

                changed = True

            elif x in ["p", "pi"]:
                if x == "pi":
                    invert = True
                else:
                    invert = False

                if len(split) > 1:
                    use_cuts = np.array(split[1:]).astype(np.int32)
                else:
                    use_cuts = None

                self.apply_cuts(invert=invert, use_cuts=use_cuts)
                self.cutplot()

            elif x == "r":
                if len(split) < 2:
                    print("Select the index of the cut to remove")
                    ind = _cli_input(mode="integer")
                else:
                    ind = split[1]

                print(f"Removing cut {int(ind)}")
                self.current_subset.remove_cut(int(ind))
                self.apply_cuts()
                self.cutplot()

                changed = True

            elif x == "subset":
                if len(split) < 2:
                    print(
                        "Select the index of the subset to switch to, or "
                        "enter 'new' to create a new subset."
                    )
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]

                if ind == "new":
                    ind = len(self.subsets)
                    print(f"Creating a new subset, index {ind}")
                    self.add_subset()

                print(f"Selecting subset {ind}")
                self.select_subset(int(ind))
                self.apply_cuts()
                self.cutplot()

                changed = True

            else:
                print(f"Invalid input: {x}")

        return changed

    def plot(
        self,
        axes=("X", "Y"),
        log=False,
        clear=False,
        xrange=None,
        yrange=None,
        zrange=None,
        show=True,
        trim=True,
        figax=None,
    ):
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

        fontsize = 16
        ticksize = 14

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

        if axes[0:2] == ("X", "Y"):
            ax.set_aspect("equal")

        if len(axes) == 3:
            ztitle = axes[2]
            title = f"{axes[0]}, {axes[1]}, {axes[2]}"
        else:
            ztitle = "# Tracks"
            title = f"{axes[0]}, {axes[1]}"

        arr[arr == 0] = np.nan

        # Calculate bounds
        if xrange[0] is None:
            xrange[0] = np.nanmin(xax)
        if xrange[1] is None:
            xrange[1] = np.nanmax(xax)
        if yrange[0] is None:
            yrange[0] = np.nanmin(yax)
        if yrange[1] is None:
            yrange[1] = np.nanmax(yax)

        if log:
            title += " (log)"
            nonzero = np.nonzero(arr)
            arr[nonzero] = np.log10(arr[nonzero])

        else:
            title += " (lin)"

        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        ax.set_xlabel(axes[0], fontsize=fontsize)
        ax.set_ylabel(axes[1], fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

        try:
            p = ax.pcolorfast(xax, yax, arr.T)

            cb_kwargs = {
                "orientation": "vertical",
                "pad": 0.07,
                "shrink": 0.8,
                "aspect": 16,
            }
            cbar = fig.colorbar(p, ax=ax, **cb_kwargs)
            cbar.set_label(ztitle, fontsize=fontsize)

        except ValueError:  # raised if one of the arrays is empty
            pass

        if show:
            plt.show()

        return fig, ax

    def cutplot(self, clear=False):

        self.cutplotfig = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        self.cutplotfig[0].subplots_adjust(hspace=0.3, wspace=0.3)

        # Figure tuple contains:
        # (fig, axarr, bkg)
        fig, axarr = self.cutplotfig

        title = f"Subset {self.current_subset_index}, "

        title += (
            f"dslice {self.current_subset.current_dslice_index} of "
            f"{self.current_subset.ndslices} selected."
        )
        fig.suptitle(title)

        # X, Y
        ax = axarr[0][0]
        self.plot(
            axes=("X", "Y"),
            show=False,
            trim=True,
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
        )

        # D, C
        ax = axarr[0][1]
        self.plot(
            axes=("D", "C"),
            show=False,
            figax=(fig, ax),
            log=True,
            trim=False,
            xrange=self.current_subset.domain.drange,
            yrange=self.current_subset.domain.crange,
        )

        # X, Y, D
        ax = axarr[1][0]
        self.plot(
            axes=("X", "Y", "D"),
            show=False,
            trim=False,
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
            zrange=self.current_subset.domain.drange,
        )

        # D, E
        ax = axarr[1][1]
        self.plot(
            axes=("D", "E"),
            show=False,
            trim=True,
            figax=(fig, ax),
            log=True,
            xrange=self.current_subset.domain.drange,
            yrange=self.current_subset.domain.erange,
        )

        plt.show()

        return fig, ax

    def focus_plot(self):
        """
        Plot the focus (z coordinate) over the scan. Used to look for
        abnormalities that may indicate a failed scan.
        """

        fig, ax = plt.subplots()

        self.plot(
            axes=("X", "Y", "Z"),
            trim=True,
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
        )

    def dslice_animation(self, savepath, fps=5, dpi=300):
        """
        Create a movie of the XY plot stepping through the dslices

        The FFMPEG library is required to save most video formats. To install,
        run

        `conda install ffmpeg`

        or

        `pip install ffmpeg`
        """
        import matplotlib.animation as animation

        print("Generating dslice animation")

        name, ext = os.path.splitext(savepath)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_aspect("equal")

        self.current_subset.select_dslice(0)
        self.apply_cuts()
        x, y, arr = self.frames()

        p = ax.pcolormesh(x, y, arr.T, cmap="plasma")

        txt = ax.text(
            0.5,
            1,
            "dslice 0",
            color="white",
            bbox={"facecolor": "k", "alpha": 0.7, "pad": 5},
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=18,
        )

        def animate(frame_number):
            txt.set_text(f"dslice {frame_number}")
            self.current_subset.select_dslice(frame_number)
            self.apply_cuts()
            x, y, arr = self.frames()

            p.set_array(arr.T.ravel())

            return (p,)

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=self.current_subset.ndslices,
            interval=1,
            blit=True,
            repeat=True,
            save_count=self.current_subset.ndslices,
        )

        if ext.lower() in [".mp4", ".avi", ".mov"]:
            try:
                writer = animation.FFMpegWriter(fps=fps)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "The ffmpeg library is required to create "
                    f"animations with file type {ext}."
                )
        elif ext.lower() in [".gif"]:
            writer = animation.PillowWriter(fps=2)
        else:
            raise ValueError(
                "No animation writer available for file extension " f"{ext}"
            )
        anim.save(savepath, writer=writer, dpi=dpi)
        plt.show()


if __name__ == "__main__":

    # data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    # data_dir = os.path.join('//expdiv','kodi','ShotData')
    # data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    data_dir = Path("C:\\", "Users", "pheu", "Data", "data_dir")

    data_path = Path(data_dir, "105350", "o105350-Ernie-PR3236-2hr_40x_s0.cpsa")

    save_path = Path(data_dir, "105350", "105350-Ernie_tracks.h5")

    obj = Scan.from_cpsa(data_path, verbose=True)

    obj.to_hdf5(save_path)

    obj.plot()
    obj.focus_plot()

    obj.cutplot()
    obj.add_cut(Cut(cmin=35))
    obj.add_cut(Cut(dmin=12))
    obj.apply_cuts()

    obj.cutplot()

    obj.to_hdf5(save_path)

    # obj.save_tracks(save_path)

    obj = Scan.from_hdf5(save_path, verbose=True)

    # domain = Cut(xmin=-5, xmax=0)
    # subset = Subset(domain=domain)

    # obj.add_subset(subset)
    # obj.add_cut(Cut(cmin=40))

    obj.cutplot()
