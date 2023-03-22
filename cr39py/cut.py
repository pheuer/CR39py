import numpy as np
import h5py
from cr39py.util.baseobject import BaseObject, identify_object


__all__ = [
    "Cut",
]


class Cut(BaseObject):
    """
    A cut is series of upper and lower bounds on tracks that should be
    excluded.

    Parameters
    ----------

    grp : h5py.Group or string file path
        An h5py Group or file path to an h5py file from which to
        load the cut


    """

    defaults = {
        "xmin": -1e6,
        "xmax": 1e6,
        "ymin": -1e6,
        "ymax": 1e6,
        "dmin": 0,
        "dmax": 1e6,
        "cmin": 0,
        "cmax": 1e6,
        "emin": 0,
        "emax": 1e6,
    }

    indices = {
        "xmin": 0,
        "xmax": 0,
        "ymin": 1,
        "ymax": 1,
        "dmin": 2,
        "dmax": 2,
        "cmin": 3,
        "cmax": 3,
        "emin": 5,
        "emax": 5,
    }

    def __init__(
        self,
        *args,
        xmin: float = None,
        xmax: float = None,
        ymin: float = None,
        ymax: float = None,
        dmin: float = None,
        dmax: float = None,
        cmin: float = None,
        cmax: float = None,
        emin: float = None,
        emax: float = None,
    ):

        super().__init__()

        _exportable_attributes = ["bounds"]
        self._exportable_attributes += _exportable_attributes

        self.bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "dmin": dmin,
            "dmax": dmax,
            "cmin": cmin,
            "cmax": cmax,
            "emin": emin,
            "emax": emax,
        }

    def __getattr__(self, key):

        if key in self.bounds.keys():
            if self.bounds[key] is None:
                return self.defaults[key]
            else:
                return self.bounds[key]
        else:
            raise ValueError(f"Unknown attribute for Cut: {key}")

    # These range properties are used to set the range for plotting
    @property
    def xrange(self):
        return [self.bounds["xmin"], self.bounds["xmax"]]

    @property
    def yrange(self):
        return [self.bounds["ymin"], self.bounds["ymax"]]

    @property
    def drange(self):
        return [self.bounds["dmin"], self.bounds["dmax"]]

    @property
    def crange(self):
        return [self.bounds["cmin"], self.bounds["cmax"]]

    @property
    def erange(self):
        return [self.bounds["emin"], self.bounds["emax"]]

    def __str__(self):
        s = [f"{key}:{val}" for key, val in self.bounds.items() if val is not None]
        s = ", ".join(s)
        if s == "":
            return "[Empty cut]"
        else:
            return s

    def test(self, trackdata):
        """
        Given tracks, return a boolean array representing which tracks
        fall within this cut
        """
        ntracks, _ = trackdata.shape
        keep = np.ones(ntracks).astype("bool")

        for key in self.bounds.keys():
            if self.bounds[key] is not None:
                i = self.indices[key]
                if "min" in key:
                    keep *= np.greater(trackdata[:, i], getattr(self, key))
                else:
                    keep *= np.less(trackdata[:, i], getattr(self, key))

        # Return a 1 for every track that is in the cut
        return keep.astype(bool)
