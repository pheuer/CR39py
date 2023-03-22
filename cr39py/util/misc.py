import os
import datetime
import h5py
import numpy as np


__all__ = [
    "decade_str",
    "remove_outliers",
    "timestamp",
    "_compressed",
    "find_file",
    "find_folder",
]


def decade_str(shot_number):
    """
    This string is used for directory names in some places
    """
    return str(shot_number)[:-4] + "xxxx"


def remove_outliers(arr, std, center="median"):
    """
    Remove outliers greater than std from the center (mean or median)
    of an array

    """

    if center == "median":
        center = np.median(arr)
    elif center == "mean":
        center = np.mean(arr)
    else:
        raise ValueError(f"Invalid keyword center: {center}")

    return arr[np.abs(arr - center) < std * np.std(arr)]


def timestamp():
    """
    Creates a timestamp string
    """
    now = datetime.datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def _compressed(*args, chunk=None, max_size=None):
    """
    Returns a sparse sampling of the data

    Parameters
    ----------

    chunk : int
        Chunk size


    max_size : int
        Max size for the largest axis. Used to estimate the chunk size.
        If the array already fits within the max size,

    """
    xaxis = None
    yaxis = None
    data = None
    if len(args) == 1:
        data = args[0]
    elif len(args) == 2:
        xaxis = args[0]
        data = args[1]
    elif len(args) == 3:
        xaxis = args[0]
        yaxis = args[1]
        data = args[2]
    else:
        raise ValueError(f"Invalid number of arguments: {len(args)}")

    # TODO: add an option to take the mean of each chunk rather than just
    # sparse sample it?

    if chunk is not None and max_size is not None:
        raise ValueError("Specify either chunk_size or max_size, but not" "both.")

    if chunk is None:
        if max_size is None:
            raise ValueError("Specify either chunk or max_size")
        else:
            chunk = int(np.max(data.shape) / max_size)

    # If the array is less than twice the desired size, return
    # the array unchanged
    if chunk < 2:
        chunk = 1
    else:
        xaxis = xaxis[::chunk]
        yaxis = yaxis[::chunk]

        if data.ndim == 1:
            data = data[::chunk]
        elif data.ndim == 2:
            data = data[::chunk, ::chunk]

    if len(args) == 1:
        return data, chunk
    elif len(args) == 2:
        return xaxis, data, chunk
    elif len(args) == 3:
        return xaxis, yaxis, data, chunk


def find_file(dir, matchstr):
    # Find all the files in that directory
    files = [x[2] for x in os.walk(dir)][0]

    # FInd ones that match the reconstruction h5 pattern
    files = [x for x in files if all(s in x for s in matchstr)]

    if len(files) == 0:
        raise ValueError(f"No file found matching {matchstr} in {dir}")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found matching {matchstr} in {dir}")
    else:
        file = files[0]

    return os.path.join(dir, file)


def find_folder(dir, matchstr):
    """
    Find subfolder
    """

    # Find all subdirectories that match the name
    dirs = [x[0] for x in os.walk(dir) if all(s in x[0] for s in matchstr)]

    if len(dirs) == 0:
        raise ValueError(f"No folder found matching {matchstr} in {dir}")
    elif len(dirs) > 1:
        raise ValueError(
            f"Multiple reconstruction folders found matching {matchstr} in {dir}"
        )
    else:
        folder = dirs[0]

    return folder
