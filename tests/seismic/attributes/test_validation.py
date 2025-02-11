"""Test Data Validation"""

import os
import inspect
import unittest

from pytest import fixture
from parameterized import parameterized_class

from dasf.utils.funcs import is_gpu_supported

from dasf_seismic import attributes
from dasf_seismic.attributes.frequency import *
from dasf_seismic.attributes.texture import *


def parameterize_all_methods_attributes():
    classes = []

    for name, obj in inspect.getmembers(attributes):
        if inspect.ismodule(obj):
            for attr_name, attr in inspect.getmembers(obj):
                if inspect.isclass(attr):
                    if attr_name != "Transform":
                        classes.append(attr_name)

    obj_attrs = []
    for cls in classes:
        try:
            obj_default = eval(cls)()
            obj_attrs.append({"obj": obj_default, "obj_name": cls})
        except TypeError:
            # We skip Enum's for example
            pass

    return obj_attrs


@unittest.skip('Validation Test is still unstable')
@parameterized_class(parameterize_all_methods_attributes())
class TestValidation(unittest.TestCase):
    @fixture(autouse=True)
    def data_dir(self, request):
        filename = request.module.__file__
        self.test_dir, _ = os.path.splitext(filename)

    def test_validation_check_from_numpy_array(self):
        raw_path = os.path.join(self.test_dir, "validation",
                                "Raw.npy")
        val_path = os.path.join(self.test_dir, "validation",
                                self.obj_name + ".npy")

        if not os.path.exists(val_path):
            raise self.skipTest("%s file does not exist" % val_path)

        in_data = np.load(raw_path)
        out_val = np.load(val_path)

        try:
            out_data = self.obj._transform_cpu(in_data)
        except NotImplementedError as nie:
            raise self.skipTest("%s: %s" % (self.obj_name, str(nie)))

        np.testing.assert_almost_equal(out_val, out_data, decimal=5)

    def test_validation_check_from_dask_numpy_array(self):
        raw_path = os.path.join(self.test_dir, "validation",
                                "Raw.npy")
        val_path = os.path.join(self.test_dir, "validation",
                                self.obj_name + ".npy")

        if not os.path.exists(val_path):
            raise self.skipTest("%s file does not exist" % val_path)

        in_data = np.load(raw_path)
        out_val = np.load(val_path)

        # Some attributes are calculated using traces
        in_shape_chunks = (10, 10, in_data.shape[2])

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        try:
            out_data = self.obj._lazy_transform_cpu(in_data).compute()
        except NotImplementedError as nie:
            raise self.skipTest("%s: %s" % (self.obj_name, str(nie)))

        np.testing.assert_almost_equal(out_val, out_data, decimal=5)
