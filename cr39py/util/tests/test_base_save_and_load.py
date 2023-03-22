# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:31:08 2022

@author: pvheu
"""

import numpy as np
from pathlib import Path

from cr39py.util.units import unit_registry as u
from cr39py.util.baseobject import BaseObject


class ExampleBaseObject(BaseObject):
    def __init__(self):
        super().__init__()

        _exportable_attributes = [
            "test_string",
            "test_int",
            "test_int_array",
            "test_float",
            "test_float_array",
            "test_bool",
            "test_bool_array",
            "test_pint_measurement",
            "test_pint_quantity",
            "test_list",
            "test_dict",
            "test_numpy_float32",
            "test_baseobject",
            "test_property",
        ]

        self._exportable_attributes += _exportable_attributes

        self.test_string = "test_str"
        self.test_int = 32
        self.test_int_array = np.array([1, 2, 3])
        self.test_float = 120.1
        self.test_float_array = np.array([1.1, 2.2, 3.3])
        self.test_bool = True
        self.test_bool_array = np.array([True, False])
        self.test_pint_measurement = u.Measurement(1.0, 1.0, u.cm)
        self.test_pint_quantity = u.Quantity(1.0, u.cm)

        self.test_list = [1, "two", 3.0]
        self.test_dict = {"one": 1, 2: "two", 3: 3.0}

        self.test_numpy_float32 = np.float32(1.0)

        self.test_baseobject = BaseObject()

    @property
    def test_property(self):
        return 1


def test_load_save(tmp_path):
    tmp_path = Path(tmp_path, "test_load_save_object.h5")

    obj = ExampleBaseObject()
    obj.to_hdf5(tmp_path)

    obj2 = ExampleBaseObject.from_hdf5(tmp_path)

    for key in obj._exportable_attributes:
        att1 = getattr(obj, key)
        att2 = getattr(obj2, key)

        print(key)

        if isinstance(att1, (np.ndarray)):
            assert np.allclose(att1, att2)

        elif isinstance(att1, (u.Measurement)):
            assert att1.value.m == att2.value.m

        elif isinstance(att1, BaseObject):
            assert att1.description == att2.description

        elif isinstance(att1, list):
            for i in range(len(att1)):
                assert att1[i] == att2[i]

        elif isinstance(att1, dict):
            print(att1)
            print(att2)
            for key in att1.keys():
                assert att1[key] == att2[key]

        else:
            assert att1 == att2


if __name__ == "__main__":

    tmp_path = Path().absolute()

    test_load_save(tmp_path)
