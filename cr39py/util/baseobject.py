__all__ = [
    "identify_object",
    "BaseObject",
]


import numpy as np
import h5py
from abc import ABC

from pathlib import Path
import datetime

from xarray import DataArray, Dataset, load_dataset

from cr39py.util.units import unit_registry as u
from cr39py.util.pkg import get_class


"""
Notes on the loading/saving functions
-------------------------------------


The following helper functions exist to standardize reading and writing of
quantities and objects to HDF5 files through h5py and/or built in methods
of the objects. We treat two kinds of objects:

quantities: str, float, int, np.ndarra, u.Quantity, u.Measurement
baseobjects : subclasses of BaseObject

objects : either baseobjects or quantities


save_quantity and load_quantity save and load quantities from an open
or unopened HDF5 file (file is opened if necessary). The open_file decorator
handles the file opening.

save_baseobject and load_baseobject save and load base object instances from
an UNOPENED HDF5 file.

save_object and load_object can take either a quantity or a baseobject
and will call the appropriate methods

"""


def identify_object(arg, group=None):
    """
    Try to identify what type of file a file or a group within an hdf5 file
    is based on  its contents and/or extension.

    All of the objects in this package save a 'class' attribute to the root
    directory of their group
    """
    if group is None:
        group = "/"

    def identify_hdf_grp(grp):
        if "class" in grp.attrs.keys():
            return str(grp.attrs["class"])

        if "PSL_per_px" in grp.keys():
            return "OMEGA IP"
        return None

    # If the provided path is actually within an HDF5 file, identify the
    # type of the group instead of the file.
    if isinstance(arg, (h5py.File, h5py.Group)):
        return identify_hdf_grp(arg)

    elif issubclass(arg.__class__, BaseObject):
        return arg.__class__.__name__

    elif isinstance(arg, (str, Path)):
        if isinstance(arg, str):
            arg = Path(arg, group=group)

        ext = arg.suffix

        if ext.lower() in [".h5", ".hdf5"]:
            with h5py.File(arg, "r") as f:
                grp = f[group]
                grptype = identify_hdf_grp(grp)
            return grptype

        elif ext.lower() in [".hdf"]:
            return "hdf4"

        elif ext.lower() in [".cpsa"]:
            return "cpsa"

        else:
            raise ValueError(f"Unrecognized file extension: {ext} in path {arg}")
    else:
        raise ValueError(f"Unrecognized object type: {type(arg)}")


class BaseObject(ABC):
    """
    An object with properties that are savable to an HDF5 file or a group
    within an HDF5 file.

    The _save and _load methods need to be extended in child classes
    to define exactly what is being loaded and/or saved.


    Paramters
    ---------

    path : str (optional)

    group : str (optional)
        The group within the h5 file where this is stored.
        Defaults to the root group

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        _exportable_attributes = ["description"]
        self._exportable_attributes = _exportable_attributes

        self.description = "BaseObject"

    @classmethod
    def from_hdf5(cls, path: [Path, str], *args, group=None, **kwargs):
        """
        Loads an instance of this class from file
        """
        if group is None:
            group = "/"

        if isinstance(path, (str, Path)):
            path = Path(path)

            objtype = identify_object(path, group=group)

            if objtype == cls.__name__:
                obj = cls()

                obj._read_hdf5_contents(path, group=group)
            else:
                raise ValueError(
                    f"Type of object {objtype} does not match "
                    "the type of this object "
                    f"{cls.__name__}"
                )
        else:
            raise ValueError(
                "path argument must be either a str or a Path, " f"not {type(path)}"
            )

        return obj

    def __str__(self):
        s = f"{self.__class__.__name__}"
        if self.description is not None:
            s += f": {self.description}"
        return s

    @staticmethod
    def _write_hdf5_entry(path, key, obj, attrs=None, group=None):
        """
        Save a quantity (number, quantity, str) into an hdf5 file

        Parameters
        ----------
        grp : h5py.Group
            Location to save the quantity


        OR (via decorator)

        path : str, Path
            Path to file in which to save the quantity

        group : str (keyword arg)
            Group in the file in which to save the quantity



        Parameters (any case)
        ---------------------

        key : str
            Name for the dataset in which to save the quantity

        obj : number, quantity, string
            The item to save

        atrrs : dict
            Dictioanry of simple quantities (str, int, float) that will be
            attached as attributes of the quantity

        Notes
        -----
        If obj is None, skip saving.

        """
        if obj is None:
            return None

        if group is None:
            group = "/"

        if attrs is None:
            attrs = {}

        # If key already exists, delete the entry
        with h5py.File(path, "a") as f:
            grp = f[group]
            if key in grp.keys():
                del grp[key]
                
        # ********************************************************************
        # Handle saving with file closed
        # ********************************************************************
        if isinstance(obj, BaseObject):
            obj._write_hdf5_contents(path, group=group + f"/{key}")
            return True

        elif isinstance(obj, (Dataset, DataArray)):
            group += f"/{key}"

            if isinstance(obj, DataArray):
                # Only DataSet can be saved, so convert it
                # It needs a name, so we give it one
                ds = obj.to_dataset(name="data")
                dtype = "xarray.DataArray"
            else:
                ds = obj
                dtype = "xarray.Dataset"

            # Format NETCDF4 and engine h5netcdf required to support
            # saving into a group and into a file with other contents
            # mode = a required to append to file rather than writing over
            # the previous contents!
            ds.to_netcdf(
                str(path),
                group=str(group),
                format="NETCDF4",
                mode="a",
                engine="h5netcdf",
                invalid_netcdf=True,
            )

            with h5py.File(path, "a") as f:
                datagrp = f[group]
                datagrp.attrs["type"] = dtype
            return True

        elif isinstance(obj, list):
            listgrp = group + f"/{key}"

            # Create the group
            with h5py.File(path, "a") as f:
                f.require_group(listgrp)
                f[listgrp].attrs["type"] = "list"

            # Save all entries into the group
            for i, item in enumerate(obj):
                BaseObject._write_hdf5_entry(path, f"item{i}", item, group=listgrp)

            return True

        elif isinstance(obj, dict):
            dictgrp = group + f"/{key}"

            # Create the group
            with h5py.File(path, "a") as f:
                f.require_group(dictgrp)
                f[dictgrp].attrs["type"] = "dict"

            # Save all entries into the group
            for key, val in obj.items():

                # Save the key and value both inside a group named with the
                # key
                BaseObject._write_hdf5_entry(path, f"{key}/key/", key, group=dictgrp)

                BaseObject._write_hdf5_entry(path, f"{key}/value/", val, group=dictgrp)

            return True

        # ********************************************************************
        # Handle saving with file open
        # ********************************************************************
        with h5py.File(path, "a") as f:
            grp = f.require_group(group)

            if key in grp.keys():
                del grp[key]

            if isinstance(obj, u.Measurement):
                grp[key] = obj.m.n
                grp[key].attrs["error"] = obj.m.s
                grp[key].attrs["unit"] = str(obj.u)
                grp[key].attrs["type"] = "pint.Measurement"

            elif isinstance(obj, u.Quantity):
                grp[key] = obj.m
                grp[key].attrs["unit"] = str(obj.u)
                grp[key].attrs["type"] = "pint.Quantity"

            elif type(obj).__module__ == "numpy":
                grp[key] = obj
                grp[key].attrs["type"] = f"numpy.{type(obj).__name__}"

            elif isinstance(obj, (bool, float, int)):
                grp[key] = obj
                grp[key].attrs["type"] = str(obj.__class__.__name__)

            # Handle string separately because string variables in hdf5 can't have
            # attributes??
            elif isinstance(obj, (str)):
                grp[key] = obj
                grp[key].attrs["type"] = "str"

            else:
                raise ValueError(f"Unknown object type {type(obj)}.")

            # Write any additional attributes provided
            if attrs is not None:
                for key, val in attrs.items():
                    grp.attrs[key] = val

    @staticmethod
    def _read_hdf5_entry(path, key, group=None, classobj=None):
        """

        Parameters
        ----------
        grp : h5py.Group
            Group that contains the dataset to be loaded

        key : str
            Key for the dataset to load

        objtype : TYPE, optional
            Type of object to try and load the data as: np.ndarray, pint.Quantity,
            or pint.Measurement.

           If not provided, attempt to guess based on dataset properties.


        Returns
        -------

        q : number, quantity, string, object
            The dataset loaded as whatever type is chosen.


        If no dataset cooresponding to the key exists, returns None

        """

        if group is None:
            group = "/"

        # First identify if the object exists
        with h5py.File(path, "r") as f:
            grp = f[group]

            # Return None if the objet doesn't exist
            if not key in grp.keys():
                return None

            # If 'type' attribute is not set, the object cannot be loaded.
            if "type" not in grp[key].attrs.keys():
                return None

            # Determine if the object is a BaseObject or something else
            # based on the type attribute
            dtype = grp[key].attrs["type"]

            # If object has type BaseObject, find the corresponding class
            # unless one is already provided
            if dtype == "BaseObject" and classobj is None:
                class_name = grp[key].attrs["class"]
                classobj = get_class(class_name)

        # If classobj is a base object, load from file using the base object
        # load method
        if classobj is not None and issubclass(classobj, BaseObject):
            return classobj.from_hdf5(path, group=group + f"/{key}")

        # If dtype is an xarray dataset, load it
        elif dtype in ["xarray.DataArray", "xarray.Dataset"]:
            group += f"/{key}"

            # Load the dataset from the hdf5 file
            # Note that load_dataset is used instead of open_dataset: the latter
            # lazy opens the file and doesn't close it!
            val = load_dataset(
                str(path), group=str(group), format="NETCDF4", engine="h5netcdf"
            )

            # If object was originally a DataArray, convert back
            # assuming name is `data`
            #
            # Use this syntax instead of .to_array() because the latter
            # also stores the data array as a dimension??
            if dtype == "xarray.DataArray":
                val = val["data"]

            return val

        elif dtype == "list":
            group += f"/{key}"

            with h5py.File(path, "r") as f:
                keys = list(f[group].keys())

            val = []
            for _key in keys:
                val.append(BaseObject._read_hdf5_entry(path, _key, group=group))
            return val

        elif dtype == "dict":
            group += f"/{key}"

            with h5py.File(path, "r") as f:
                keys = list(f[group].keys())

            val = {}
            for _key in keys:

                # Load the key (as saved, not by the folder name)
                thiskey = BaseObject._read_hdf5_entry(
                    path, "key", group=group + f"/{_key}"
                )
                # Then load the value
                thisval = BaseObject._read_hdf5_entry(
                    path, "value", group=group + f"/{_key}"
                )
                # Assemble into new dict
                val[thiskey] = thisval

            return val

        # Otherwise, load quantity based on the stored dtype
        else:
            with h5py.File(path, "r") as f:
                grp = f[group]
                dtype = grp[key].attrs["type"]

                if dtype == "pint.Quantity":
                    val = grp[key][...]
                    unit = grp[key].attrs["unit"]
                    val = u.Quantity(val, unit)

                elif dtype == "pint.Measurement":
                    val = grp[key][...]
                    unit = grp[key].attrs["unit"]
                    error = grp[key].attrs["error"]
                    val = u.Measurement(val, error, unit)

                elif "numpy" in dtype:
                    np_dtype = dtype[6:]
                    val = grp[key][...]
                    if np_dtype == "ndarray":
                        pass
                    else:
                        val = val.astype(np_dtype)

                elif dtype == "int":
                    val = int(grp[key][...])

                elif dtype == "float":
                    val = float(grp[key][...])

                elif dtype == "bool":
                    val = bool(grp[key][...])

                elif dtype == "str":
                    val = grp[key][()].decode("utf-8")

                else:
                    raise ValueError(f"Unrecognized dtype {dtype} for item {key}")

            # If classobj is set, pass the output through that prior to returning
            if classobj is None:
                return val
            else:
                return classobj(val)

    def to_hdf5(self, path, group=None):
        """
        Save this object to an h5 file or group within an h5 file
        """

        path = Path(path)

        if group is None:
            group = "/"

        tmp_path = Path(path.parent, path.stem + "_tmp" + path.suffix)

        tmp_path.unlink(missing_ok=True)

        self._write_hdf5_contents(tmp_path, group=group)

        # If successful to this point, delete the existing file and rename
        # the temp file to replace it
        path.unlink(missing_ok=True)
        tmp_path.rename(path)

    def _write_hdf5_contents(self, path, group=None):
        if group is None:
            group = "/"

        # Prepare the group
        with h5py.File(path, "a") as f:
            grp = f.require_group(group)

            # Empty the group before saving new data there
            for key in grp.keys():
                del grp[key]

            grp.attrs["type"] = "BaseObject"
            grp.attrs["class"] = self.__class__.__name__
            grp.attrs["timestamp"] = datetime.datetime.now().isoformat()

        # Save all exportable attributes into the group
        for name in self._exportable_attributes:
            if hasattr(self, name):
                BaseObject._write_hdf5_entry(
                    path, name, getattr(self, name), group=group
                )

    def _read_hdf5_contents(self, path, group=None):
        if group is None:
            group = "/"

        with h5py.File(path, "r") as f:
            grp = f[group]
            keys = list(grp.keys())

        for key in keys:
            val = self._read_hdf5_entry(path, key, group=group)

            # Try to set the attribute
            # If it doesn't work, it's probably a property in which case
            # we will skip setting it
            try:
                setattr(self, key, val)
            except AttributeError:
                pass
