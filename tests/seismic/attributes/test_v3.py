"""V3 validation test suite"""
import os
import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except Exception:
    pass

from dasf.utils.funcs import is_gpu_supported
from pytest import fixture
from parameterized import parameterized_class

from params_texture import attributes as texture

from utils import get_class_name, get_item


@parameterized_class(
    texture, # currently only texture attributes (GLCM) have V3 testing support
    class_name_func=get_class_name,
)
class TestV3(unittest.TestCase):
    """Base class for V3 validation tests"""

    # Default Test params
    operator_params = {}
    v3_operator_params = {}
    v3_in_shape_chunks = (10, 10, -1)
    v3_skip = False
    v3_remove_border = False
    v3_input = None
    v3_expected = None

    @fixture(autouse=True)
    def input_data(self, request):
        filename = request.module.__file__
        test_dir = os.path.split(filename)[:-1]
        self.test_dir = os.path.join(*test_dir)

    def test_lazy_implementation_cpu(self):
        """Test result values of CPU lazy implementation using external library result as ground truth"""
        operator_params = self.v3_operator_params if self.v3_operator_params != {} else self.operator_params # Some attributes have specific params for V3 testing, while others use the generic params
        operator = self.operator_cls(**operator_params)

        if self.v3_skip:
            raise self.skipTest(f"{operator.__class__.__name__}: SKIP")
        
        assert self.v3_input is not None
        assert self.v3_expected is not None
        input_path = os.path.join(
            self.test_dir, "test_v3_data", self.v3_input
        )
        in_data = np.load(input_path).astype("float64")
        expected_path = os.path.join(
            self.test_dir, "test_v3_data", self.v3_expected
        )
        expected = np.load(expected_path).astype("float64")

        in_data_cpu = [in_data]
        in_data_lazy_cpu = [
            da.from_array(data, chunks=get_item(self.v3_in_shape_chunks, i))
            for i, data in enumerate(in_data_cpu)
        ]
        try:
            out_data_lazy_cpu = operator._lazy_transform_cpu(*in_data_lazy_cpu)
        except NotImplementedError as nie:
            raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

        output_comp = out_data_lazy_cpu.compute()
        if self.v3_remove_border is not None:
            expected = expected[self.v3_remove_border]
            output_comp = output_comp[self.v3_remove_border]
        np.testing.assert_almost_equal(
            expected,
            output_comp,
            decimal=5,
            err_msg="DASK CPU x Expected Comparison",
        )

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_lazy_implementation_gpu(self):
        """Test result values of GPU lazy implementation using external library result as ground truth"""
        operator_params = self.v3_operator_params if self.v3_operator_params != {} else self.operator_params # Some attributes have specific params for V3 testing, while others use the generic params
        operator = self.operator_cls(**operator_params)

        if self.v3_skip:
            raise self.skipTest(f"{operator.__class__.__name__}: SKIP")
        
        assert self.v3_input is not None
        assert self.v3_expected is not None
        input_path = os.path.join(
            self.test_dir, "test_v3_data", self.v3_input
        )
        in_data = cp.load(input_path).astype("float64")
        expected_path = os.path.join(
            self.test_dir, "test_v3_data", self.v3_expected
        )
        expected = np.load(expected_path).astype("float64")

        in_data_gpu = [in_data]
        in_data_lazy_gpu = [
            da.from_array(data, chunks=get_item(self.v3_in_shape_chunks, i))
            for i, data in enumerate(in_data_gpu)
        ]
        try:
            out_data_lazy_cpu = operator._lazy_transform_gpu(*in_data_lazy_gpu)
        except NotImplementedError as nie:
            raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

        output_comp = out_data_lazy_cpu.compute().get()
        if self.v3_remove_border is not None:
            expected = expected[self.v3_remove_border]
            output_comp = output_comp[self.v3_remove_border]
        np.testing.assert_almost_equal(
            expected,
            output_comp,
            decimal=5,
            err_msg="DASK GPU x Expected Comparison",
        )
            
