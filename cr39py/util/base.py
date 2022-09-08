# -*- coding: utf-8 -*-
"""
Define the base saveable object class
"""

import h5py
from abc import ABC

class BaseObject(ABC):
    """
    An object with properties that are savable to an HDF5 file or a group
    within an HDF5 file.
    
    The _save and _load methods need to be extended in child classes
    to define exactly what is being loaded and/or saved.
    
    """
    
    def __init__(self):
        # The path to the file
        self.path = None
        # The group within the h5 file where this is stored.
        # Defaults to the root group
        self.group = '/'
    
    
    def _save(self, grp):
        """
        Save this object to an h5 group
        
        
        Subclasses should call this method at the begining of their own
        _save method.
        """
        
        # Empty the group before saving new data there
        for key in grp.keys():
            del grp[key]
        
        grp.attrs['class'] = self.__class__.__name__
        
        
    def _load(self, grp):
        """
        Load an object from an h5 group
        """
        pass
    
    
    def save(self, path, group=None):
        """
        Save this object to an h5 file or group within an h5 file
        """ 
        
        if isinstance(path, h5py.File):
            self.path = path.filename
            self.group = path.name
            self._save(path)
            
        elif isinstance(path, h5py.Group):
            self.path = path.file.filename
            self.group = path.name
            self._save(path)
                
        else:
            self.path = path
            self.group = '/'
            with h5py.File(self.path, 'a') as f:
                if group is not None:
                    grp = f[group]
                else:
                    grp = f
                
                self._save(grp)


    def load(self, path, group='/'):
        """
        Load this object from a file
        """
        
        if isinstance(path, h5py.File):
            self.path = path.filename
            self.group = path.name
            self._load(path)
            
        elif isinstance(path, h5py.Group):
            self.path = path.file.filename
            self.group = path.name
            self._load(path)
            
        else:
            self.path = path
            self.group = group
        
            with h5py.File(self.path, 'r') as f:
                
                if group is not None:
                    grp = f[group]
                else:
                    grp = f
                
                self._load(grp)