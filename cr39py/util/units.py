"""
Temporary stand-in for lotus unit registry
"""

import numpy as np
import pint
import warnings

unit_registry = pint.UnitRegistry()

# PSL as a dimensionless unit
unit_registry.define("PSL=")


__all__ = []


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unit_registry.Quantity([])
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pint")
unit_registry.enable_contexts("boltzmann")
unit_registry.default_format = "0.8g~P"
unit_registry.default_system = "cgs"

u = unit_registry
