from cr39py.util.baseobject import BaseObject
from cr39py.cut import Cut

__all__ = [
    "Subset",
]


class Subset(BaseObject):
    """
    A subset of the track data. The subset is defined by a domain, a list
    of cuts, and a number of diameter slices (Dslices).

    Paramters
    ---------

    grp : h5py.Group or string file path
        An h5py Group or file path to an h5py file from which to
        load the subset


    Notes
    -----

    Domain
    ------
    The domain is the area in parameter space the subset encompasses. This
    could limit the subset to a region in space (e.g. x=[-5, 0]) or another
    parameter (e.g. D=[0,20]). The domain is represented by a cut, but it is
    an inclusive rather than an exclusive cut.

    List of Cuts
    ------------
    The subset includes a list of cuts that are used to exclude tracks that
    would otherwise be included in the domain.

    DSlices
    ------
    Track diameter is proportional to particle energy, so slicing the subset
    into bins sorted by diameter sorts the tracks by diameter. Slices are
    created by equally partitioining the tracks in the subset into some
    number of dslices.



    Subset is a little different than other GroupBaseObject instances. In this
    case, ndslices is usually set (which determines the length of the dslice list)
    and then datasets are calculated in Scan and added using set_dslice()
    rather than add_dslice()


    """

    def __init__(self, *args, domain=None, ndslices=None):
        super().__init__()

        _exportable_attributes = ["domain", "cuts", "current_dslice_index", "ndslices"]
        self._exportable_attributes += _exportable_attributes

        self.cuts = []
        self.ndslices = None

        if domain is not None:
            self.set_domain(domain)
        # If no domain is set, set it with an empty cut
        else:
            self.domain = Cut()

        self.current_dslice_index = 0

        self.ndslices = None

        # By default, set the number of dslices to be 1
        if ndslices is None:
            self.set_ndslices(1)
        else:
            self.set_ndslices(ndslices)

    def __str__(self):
        s = ""
        s += "Domain:" + str(self.domain) + "\n"
        s += "Current cuts:\n"
        if len(self.cuts) == 0:
            s += "No cuts set yet\n"
        else:
            for i, cut in enumerate(self.cuts):
                s += f"Cut {i}: {str(cut)}\n"
        s += f"Num. dslices: {self.ndslices} "
        s += f"[Selected dslice index: {self.current_dslice_index}]\n"

        return s

    def set_domain(self, cut):
        """
        Sets the domain cut: an inclusive cut that will not be inverted
        """
        self.domain = cut

    def select_dslice(self, i):
        if i is None:
            self.current_dslice_index = 0
        elif i > self.ndslices - 1:
            print(
                f"Cannot select the {i} dslice, there are only "
                f"{self.ndslices} dslices."
            )
        else:
            self.current_dslice_index = i

    def set_ndslices(self, ndslices):
        """
        Sets the number of ndslices
        """
        if not isinstance(ndslices, int) or ndslices < 0:
            print(
                "ndslices must be an integer > 0, but the provided value"
                f"was {ndslices}"
            )

        else:
            self.ndslices = int(ndslices)

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    @property
    def ncuts(self):
        return len(self.cuts)

    def add_cut(self, c):
        self.cuts.append(c)

    def remove_cut(self, i):
        if i > len(self.cuts) - 1:
            print(
                f"Cannot remove the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts.pop(i)

    def replace_cut(self, i, c):
        if i > len(self.cuts) - 1:
            print(
                f"Cannot replace the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts[i] = c


if __name__ == "__main__":
    import os

    domain = Cut(xmin=-5, xmax=0)
    c1 = Cut(dmax=10)
    c2 = Cut(cmax=40)
    s = Subset(domain=domain)
    s.set_ndslices(5)
    s.select_dslice(2)
    s.add_cut(c1)
    s.add_cut(c2)

    path = os.path.join(os.getcwd(), "testsubset.h5")
    print(path)
    s.to_hdf5(path)

    s2 = Subset.from_hdf5(path)
    print(s2.domain.bounds)
    print(s2.cuts[0].bounds)
    print(s2.cuts[1].bounds)
