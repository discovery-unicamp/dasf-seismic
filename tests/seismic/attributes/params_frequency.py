"""Frequency attributes testing parameters"""

from dasf_seismic.attributes.frequency import (
    RGBBlending,
)


attributes = [
    {
        "operator_cls": RGBBlending,
        "dtypes": {
            "float32": "int64",
            "float64": "int64",
        },
        "v1_in_shape": (3, 100, 100, 100),
        "v1_in_shape_chunks": (3, 50, 50, 50),
        "v1_expected_shape": (100, 100, 100),
        "v2_outer_dim": 3,
        "v2_in_shape_chunks": (3, 50, 50, 50),
        "v3_skip": True,
    },
]
