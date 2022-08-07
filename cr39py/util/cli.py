# -*- coding: utf-8 -*-
"""
Utilities for making command line interfaces (CLIs)

@author: pvheu
"""

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
            
        elif mode=='alpha-integer':
            if x in integers.union(alphas):
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