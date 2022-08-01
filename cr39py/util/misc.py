import os

def _compressed(xaxis, yaxis, data, chunk=25):
    """
    Returns a sparse sampling of the data
    """
    x_c = xaxis[::chunk]
    y_c = yaxis[::chunk]
    arr = data[::chunk, ::chunk]
    
    return x_c, y_c, arr


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
            raise ValueError(f"Multiple reconstruction folders found matching {matchstr} in {dir}")
        else:
            folder = dirs[0]
            
        return folder